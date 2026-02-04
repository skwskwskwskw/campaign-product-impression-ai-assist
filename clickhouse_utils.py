"""
ClickHouse utilities with AWS Secrets integration, connection pooling, and retry mechanisms.
"""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ParamValidationError  # noqa: F401
import json
import clickhouse_connect
import logging
from typing import Dict, Any
from functools import wraps
import time
import random


def initialize_credentials(
    secret_name="SHARED_LAMBDA_CREDENTIALS",
    region_name="ap-southeast-2",
    timeout=15,
    profile_name="default"
):
    """
    Initialize the global credentials by calling get_secret_with_retry.
    """
    global credentials
    credentials = get_secret_with_retry(secret_name, region_name, timeout, profile_name)[0]
    return credentials


def get_secret_with_retry(
    secret_name="SHARED_LAMBDA_CREDENTIALS",
    region_name="ap-southeast-2",
    timeout=15,
    profile_name="default"
):
    """
    Retrieves a secret from AWS Secrets Manager with retry logic and a timeout.
    """
    config = Config(connect_timeout=timeout, read_timeout=timeout)
    try:
        session = boto3.session.Session(profile_name=profile_name, region_name=region_name)
        client = session.client(service_name='secretsmanager', region_name=region_name, config=config)

        resp = client.get_secret_value(
            SecretId=secret_name,
            VersionStage='AWSCURRENT'
        )
        secret = resp['SecretString']
        secret_dict = json.loads(secret)
        logging.info(f"Successfully retrieved secret: {secret_name}")
        return secret_dict, secret
    except Exception as e:
        logging.warning(f"Failed to retrieve secret with profile '{profile_name}': {e}")
        # Fallback to default session (e.g. Lambda)
        session = boto3.session.Session(region_name=region_name)
        client = session.client(service_name='secretsmanager', region_name=region_name, config=config)
        resp = client.get_secret_value(
            SecretId=secret_name,
            VersionStage='AWSCURRENT'
        )
        secret = resp['SecretString']
        secret_dict = json.loads(secret)
        logging.info(f"Successfully retrieved secret using default session: {secret_name}")
        return secret_dict, secret


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logging.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise

                    logging.warning(f"Attempt {retries} of {func.__name__} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay + random.uniform(0, 0.1))  # Add jitter
                    current_delay *= backoff

        return wrapper
    return decorator


@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def create_clickhouse_client(credentials, database='profitpeak'):
    """
    Create a ClickHouse client using credentials from Secrets Manager.
    Handles vpce.* host rewriting as fallback.
    Implements retry mechanism.
    """
    original_host = credentials['CLICKHOUSE_URL'].replace('https://', '')
    port = int(credentials['CLICKHOUSE_PORT'])
    user = credentials['CLICKHOUSE_USER']
    password = credentials['CLICKHOUSE_PASSWORD']

    try:
        client = clickhouse_connect.get_client(
            host=original_host,
            port=port,
            secure=True,
            username=user,
            verify=False,
            password=password,
            database=database
        )
        logging.info(f"Successfully connected to ClickHouse: {original_host}")
        return client
    except Exception as e:
        logging.warning(f"First attempt failed with host='{original_host}': {e}")
        alternative_host = original_host.replace('vpce.', '')
        if alternative_host != original_host:
            logging.info(f"Trying again after removing 'vpce.': {alternative_host}")
            try:
                client = clickhouse_connect.get_client(
                    host=alternative_host,
                    port=port,
                    secure=True,
                    username=user,
                    verify=False,
                    password=password,
                    database=database
                )
                logging.info(f"Successfully connected to ClickHouse: {alternative_host}")
                return client
            except Exception as e2:
                logging.error(f"Second attempt also failed with host='{alternative_host}': {e2}")
                raise
        else:
            raise


@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def create_clickhouse_client_staging(credentials, database='profitpeak'):
    """
    Create a ClickHouse client using credentials from Secrets Manager.
    Handles vpce.* host rewriting as fallback.
    Implements retry mechanism.
    """
    original_host = 'ue0yeboc3m.ap-southeast-2.aws.clickhouse.cloud'
    port = int(credentials['CLICKHOUSE_PORT'])
    user = 'default'
    password = 'VnpgEx6jKqMZ_'

    try:
        client = clickhouse_connect.get_client(
            host=original_host,
            port=port,
            secure=True,
            username=user,
            verify=False,
            password=password,
            database=database
        )
        logging.info(f"Successfully connected to staging ClickHouse: {original_host}")
        return client
    except Exception as e:
        logging.error(f"Can't connect to staging ClickHouse: {e}")
        raise


class ClickHouseClientManager:
    """
    Manages ClickHouse client connections with pooling and caching.
    """

    def __init__(self, credentials: Dict[str, Any], database: str = 'profitpeak'):
        self.credentials = credentials
        self.database = database
        self._primary_client = None
        self._staging_client = None
        self._query_cache = {}

    def get_primary_client(self):
        """Get or create the primary ClickHouse client."""
        if self._primary_client is None:
            self._primary_client = create_clickhouse_client(self.credentials, self.database)
        return self._primary_client

    def get_staging_client(self):
        """Get or create the staging ClickHouse client."""
        if self._staging_client is None:
            self._staging_client = create_clickhouse_client_staging(self.credentials, self.database)
        return self._staging_client

    def execute_query_with_retry(self, client, query: str, use_cache: bool = True):
        """
        Execute a query with retry mechanism and optional caching.

        Args:
            client: ClickHouse client
            query: SQL query to execute
            use_cache: Whether to use query result caching

        Returns:
            Query result as DataFrame
        """
        # Use cache if enabled and query exists in cache
        if use_cache and query in self._query_cache:
            logging.info(f"Returning cached result for query: {query[:50]}...")
            return self._query_cache[query]

        # Execute query with retry mechanism
        result = client.query_df(query)

        # Store in cache if enabled
        if use_cache:
            self._query_cache[query] = result

        return result

    def close_connections(self):
        """Close all client connections."""
        if self._primary_client:
            self._primary_client.close()
            self._primary_client = None
        if self._staging_client:
            self._staging_client.close()
            self._staging_client = None
        self._query_cache.clear()
        logging.info("Closed all ClickHouse connections and cleared cache")