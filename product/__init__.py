"""
Main product module that imports all functions for easy access.
"""

from .config import *
from .utils import *
from .clickhouse_utils import *
from .data_retrieval import *
from .url_processing import *
from .product_processing import *
from .metrics_coalescing import *
from .product_classification import *
from .ad_targeting import *
from .helpers import *
from .main_product_identifier import *

__all__ = [
    # Config
    'Config',
    'get_config',

    # Utils
    'get_system_workers_info',
    'get_optimal_workers',
    'setup_logging',
    'memory_monitor',
    'validate_dataframe',

    # ClickHouse utils
    'initialize_credentials',
    'get_secret_with_retry',
    'create_clickhouse_client',
    'create_clickhouse_client_staging',
    'retry_on_failure',
    'ClickHouseClientManager',

    # Data retrieval
    'get_metrics_by_country',
    'get_metrics_by_product',
    'get_raw_metrics',
    'get_latest_ads',
    'get_latest_products_and_groups',
    'get_latest_product_groups',
    'get_websites',
    'validate_input_params',

    # URL processing
    'beautify_url',
    '_find_url_in_params',
    'clean_destination_url',
    'to_list',
    'beautify_urls_parallel',
    'process_destination_urls',

    # Product processing
    'parse_collections',
    'ensure_list',
    'to_scalar',

    # Metrics coalescing
    'coalesce_products_base_country_supplement_robust',
    'METRICS',
    'FULL_GRAIN',
    'validate_metrics_dataframes',

    # Product classification
    'classify_product_id_tokens_parallel',
    'validate_product_classification_inputs',

    # Ad targeting
    'build_targeting',
    'build_product_indices',
    'match_one_signature',
    'fuzzy_match_pgids',

    # Main product identification
    'identify_main_products',
    'identify_main_products_vectorized',

    # Helpers
    'urldecode_recursive',
    'extract_ad_name',
]