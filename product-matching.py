#!/usr/bin/env python3
"""
Product Matching Script
This script implements the workflow from the understanding-prods-20260122.ipynb notebook
to generate the final result tables for product attribution and impression analysis.
"""

import importlib
import importlib.util
import gc
import os
import sys
from typing import Tuple

# Validate third-party dependencies before importing them.
pandas_spec = importlib.util.find_spec("pandas")
if pandas_spec is None:
    raise SystemExit(
        "Missing dependency: pandas. Install it with `pip install pandas` before running this script."
    )

numpy_spec = importlib.util.find_spec("numpy")
if numpy_spec is None:
    raise SystemExit(
        "Missing dependency: numpy. Install it with `pip install numpy` before running this script."
    )

pd = importlib.import_module("pandas")
np = importlib.import_module("numpy")

# Ensure the repository root is on the Python path to import local modules.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import all functions from local modules.
from product.config import get_config
from product.utils import (
    get_system_workers_info,
    get_optimal_workers,
    setup_logging,
    memory_monitor,
)
from product.clickhouse_utils import (
    initialize_credentials,
    create_clickhouse_client,
    create_clickhouse_client_staging,
)
from product.data_retrieval import (
    get_websites,
    get_latest_ads,
    get_latest_products_and_groups,
    get_latest_product_groups,
    get_raw_metrics,
    get_metrics_by_country,
    get_metrics_by_product,
)
from product.url_processing import (
    process_destination_urls,
    beautify_urls_parallel,
)
from product.product_processing import (
    ensure_list,
    to_scalar,
)
from product.metrics_coalescing import (
    coalesce_products_base_country_supplement_robust,
    METRICS,
    FULL_GRAIN,
)
from product.product_classification import classify_product_id_tokens_parallel
from product.ad_targeting import build_targeting
from product.main_product_identifier import (
    identify_main_products,
    identify_main_products_vectorized,
)
from product.helpers import (
    urldecode_recursive,
    extract_ad_name,
)


def load_or_fetch_data(
    client,
    wid: str = '6839260124a2adf314674a5e',
    start_date: str = "2025-10-01",
    end_date: str = "2025-12-31",
    parquet_dir: str = "."
) -> Tuple[pd.DataFrame, ...]:
    """
    Load data from parquet files if they exist, otherwise fetch from ClickHouse.

    Args:
        client: ClickHouse client connection
        wid: Website ID to fetch data for
        start_date: Start date for metrics queries
        end_date: End date for metrics queries
        parquet_dir: Directory to store/load parquet files

    Returns:
        Tuple of DataFrames: (websites, df_ads_ori, df_prod, df_prod_group,
                             df_metrics, df_metrics_by_country, df_metrics_by_products)
    """
    parquet_files = {
        'websites': os.path.join(parquet_dir, 'websites.parquet'),
        'df_ads_ori': os.path.join(parquet_dir, 'df_ads_ori.parquet'),
        'df_prod': os.path.join(parquet_dir, 'df_prod.parquet'),
        'df_prod_group': os.path.join(parquet_dir, 'df_prod_group.parquet'),
        'df_metrics': os.path.join(parquet_dir, 'df_metrics.parquet'),
        'df_metrics_by_country': os.path.join(parquet_dir, 'df_metrics_by_country.parquet'),
        'df_metrics_by_products': os.path.join(parquet_dir, 'df_metrics_by_products.parquet')
    }

    # Check if all parquet files exist
    all_exist = all(os.path.exists(file) for file in parquet_files.values())

    if all_exist:
        import logging
        logging.info("Loading data from existing parquet files...")
        websites = pd.read_parquet(parquet_files['websites'])
        df_ads_ori = pd.read_parquet(parquet_files['df_ads_ori'])
        df_prod = pd.read_parquet(parquet_files['df_prod'])
        df_prod_group = pd.read_parquet(parquet_files['df_prod_group'])
        df_metrics = pd.read_parquet(parquet_files['df_metrics'])
        df_metrics_by_country = pd.read_parquet(parquet_files['df_metrics_by_country'])
        df_metrics_by_products = pd.read_parquet(parquet_files['df_metrics_by_products'])
    else:
        import logging
        logging.info(f"Fetching data from ClickHouse with date range: {start_date} to {end_date}")

        # Get websites
        websites = get_websites(client)
        websites.to_parquet(parquet_files['websites'])

        # Get data for the specified website
        df_ads_ori = get_latest_ads(client, wid)
        df_prod = get_latest_products_and_groups(client, wid)
        df_prod_group = get_latest_product_groups(client, wid)

        df_ads_ori.drop(columns='insertId', errors='ignore').to_parquet(parquet_files['df_ads_ori'])
        df_prod.to_parquet(parquet_files['df_prod'])
        df_prod_group.to_parquet(parquet_files['df_prod_group'])

        df_metrics = get_raw_metrics(client, wid, start_date, end_date)
        df_metrics.drop(columns='insertId', errors='ignore').to_parquet(parquet_files['df_metrics'])
        df_metrics_by_country = get_metrics_by_country(client, wid, start_date, end_date)
        df_metrics_by_country.drop(columns='insertId', errors='ignore').to_parquet(parquet_files['df_metrics_by_country'])
        df_metrics_by_products = get_metrics_by_product(client, wid, start_date, end_date)
        df_metrics_by_products.drop(columns='insertId', errors='ignore').to_parquet(parquet_files['df_metrics_by_products'])

    return websites, df_ads_ori, df_prod, df_prod_group, df_metrics, df_metrics_by_country, df_metrics_by_products


def process_urls(df_ads_ori: pd.DataFrame, max_workers: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process destination URLs to beautify and extract insights.
    
    Args:
        df_ads_ori: Original ads DataFrame
        max_workers: Number of workers for parallel processing
        
    Returns:
        Tuple of DataFrames: (df_ads_exploded, df_ads_unique_urls, df_ads_exploded_dist)
    """
    print("Processing destination URLs...")
    df_ads_exploded, df_ads_unique_urls = process_destination_urls(
        df_ads_ori, 
        parallel=True, 
        max_workers=max_workers
    )
    
    df_ads_exploded_dist = df_ads_exploded.drop_duplicates(
        subset=["destinationUrl_original", "destinationUrl_beautified"]
    )
    
    return df_ads_exploded, df_ads_unique_urls, df_ads_exploded_dist


def process_product_groups(df_prod_group: pd.DataFrame) -> pd.DataFrame:
    """
    Process product groups to extract collections and other attributes.
    
    Args:
        df_prod_group: Product groups DataFrame
        
    Returns:
        Processed product groups DataFrame
    """
    print("Processing product groups...")
    df_prod_subset = df_prod_group[['productGroupId', 'collections', 'name', 'url']].copy()
    
    df_prod_subset["collections_list"] = df_prod_subset["collections"].apply(ensure_list)
    df_prod_subset = df_prod_subset.reset_index(drop=True)
    df_prod_subset["row_id"] = df_prod_subset.index

    df_prod_subset_exploded = (
        df_prod_subset
        .explode("collections_list", ignore_index=False)
        .rename(columns={"collections_list": "collection"})
    )

    # sanitize columns used for dedupe (must be hashable)
    df_prod_subset_exploded["collection"] = df_prod_subset_exploded["collection"].map(to_scalar)
    df_prod_subset_exploded["name"] = df_prod_subset_exploded["name"].map(to_scalar)
    df_prod_subset_exploded["url"] = df_prod_subset_exploded["url"].map(to_scalar)

    # name as a "collection" row (optional)
    df_name_as_collection = (
        df_prod_subset[['productGroupId', 'name', 'url', 'row_id']]
        .drop_duplicates(subset=['productGroupId', 'name', 'url'])
        .assign(collection=lambda d: d['name'].map(to_scalar))
    )

    df_name_as_collection["name"] = df_name_as_collection["name"].map(to_scalar)
    df_name_as_collection["url"] = df_name_as_collection["url"].map(to_scalar)

    df_prod_subset_exploded = (
        pd.concat([df_prod_subset_exploded, df_name_as_collection], ignore_index=True)
        .drop_duplicates(subset=['productGroupId', 'collection', 'name', 'url'])
    )

    # final clean trim
    df_prod_subset_exploded["collection"] = (
        df_prod_subset_exploded["collection"]
        .astype("string")
        .str.strip(" '\"")
        .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    )
    
    return df_prod_subset_exploded


def coalesce_metrics(
    df_metrics_by_products: pd.DataFrame,
    df_metrics_by_country: pd.DataFrame,
    cutoff_date: str = "2025-11-01"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coalesce product and country metrics.

    Args:
        df_metrics_by_products: Product-level metrics DataFrame
        df_metrics_by_country: Country-level metrics DataFrame
        cutoff_date: Date to filter metrics from

    Returns:
        Tuple of DataFrames: (coalesced_df, temp_df)
    """
    import logging
    logging.info(f"Coalescing metrics with cutoff date: {cutoff_date}")
    cutoff = pd.to_datetime(cutoff_date)

    coalesced_df, temp_df = coalesce_products_base_country_supplement_robust(
        df_metrics_by_products[df_metrics_by_products.timestamp >= cutoff],
        df_metrics_by_country[df_metrics_by_country.timestamp >= cutoff],
        include_country_code=False,
    )

    return coalesced_df, temp_df


def enhance_ads_data(
    df_ads_ori: pd.DataFrame,
    df_ads_exploded_dist: pd.DataFrame,
    df_prod_group: pd.DataFrame,
    max_workers: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhance ads data with processed URLs and product information.
    
    Args:
        df_ads_ori: Original ads DataFrame
        df_ads_exploded_dist: Distinct exploded ads DataFrame
        df_prod_group: Product groups DataFrame
        max_workers: Number of workers for parallel processing
        
    Returns:
        Tuple of DataFrames: (df_ads_enhanced, df_prod_group_enhanced)
    """
    print("Enhancing ads data...")
    
    # Create links between ads and processed URLs
    links = pd.merge(
        df_ads_ori[['websiteId', 'platform', 'campaignId', 'adSetId', 'adId', 'destinationUrl', 'extraParams']],
        df_ads_exploded_dist[['destinationUrl_original', 'destinationUrl_beautified']], 
        how='left', 
        left_on='destinationUrl', 
        right_on='destinationUrl_original'
    ).drop(columns='destinationUrl_original').drop_duplicates([
        'platform', 'destinationUrl', 'campaignId', 'adSetId', 'adId'
    ])

    links = links[~links.destinationUrl_beautified.isna()]

    # Decode URLs
    links['destinationUrl_beautified_decoded'] = (
        links["destinationUrl_beautified"]
          .astype("string")
          .map(lambda x: urldecode_recursive(x) if x else x)
    )

    # Extract collections and products from URLs
    s = links["destinationUrl_beautified_decoded"].astype("string").fillna("")
    links["collections_"] = (
        s.str.extract(r"/collection(?:s)?/([A-Za-z0-9_-]+)", expand=False)
         .fillna("")
    )
    links["products_"] = (
        s.str.extract(r"/products/([A-Za-z0-9_-]+)", expand=False)
         .fillna("")
    )

    # Process product groups similarly
    df_prod_group['url_beautified'] = beautify_urls_parallel(
        df_prod_group['url'], 
        max_workers=max_workers
    )
    df_prod_group['url_beautified_decoded'] = (
        df_prod_group["url_beautified"]
          .astype("string")
          .map(lambda x: urldecode_recursive(x) if x else x)
    )

    s = df_prod_group["url_beautified_decoded"].astype("string").fillna("")
    df_prod_group["collections_"] = (
        s.str.extract(r"/collection(?:s)?/([A-Za-z0-9_-]+)", expand=False)
         .fillna("")
    )
    df_prod_group["products_"] = (
        s.str.extract(r"/products/([A-Za-z0-9_-]+)", expand=False)
         .fillna("")
    )

    # Create enhanced ads data
    df_ads_enhanced = pd.merge(
        df_ads_ori, 
        links[['websiteId', 'platform', 'campaignId', 'adSetId', 'adId', 'destinationUrl', 
               'destinationUrl_beautified_decoded', 'collections_', 'products_']], 
        how='left',
        on=['websiteId', 'platform', 'campaignId', 'adSetId', 'adId', 'destinationUrl']
    )

    df_ads_enhanced["ad_name"] = df_ads_enhanced["extraParams"].apply(extract_ad_name)
    
    return df_ads_enhanced, df_prod_group


def classify_product_ids(coalesced_df: pd.DataFrame, df_prod: pd.DataFrame, max_workers: int) -> pd.DataFrame:
    """
    Classify product IDs in the coalesced DataFrame.
    
    Args:
        coalesced_df: Coalesced metrics DataFrame
        df_prod: Products DataFrame
        max_workers: Number of workers for parallel processing
        
    Returns:
        Updated coalesced DataFrame with classified product IDs
    """
    print("Classifying product IDs...")
    
    # Ensure you have a 'productId_' column as source
    if "productId_" not in coalesced_df.columns and "productId" in coalesced_df.columns:
        coalesced_df["productId_"] = coalesced_df["productId"]

    # Classify product ID tokens
    coalesced_df = classify_product_id_tokens_parallel(
        coalesced_df,
        df_prod,
        source_col="productId_",
        prefer="productGroupId",     # or "productId"
        n_workers=max_workers,
        chunk_size=10000,
        fill_group_from_product=True
    )
    
    return coalesced_df


def build_final_targeting(df_ads_enhanced: pd.DataFrame, df_prod_group: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the final targeting table.
    
    Args:
        df_ads_enhanced: Enhanced ads DataFrame
        df_prod_group: Enhanced product groups DataFrame
        
    Returns:
        Tuple of DataFrames: (final_targeting, debug_rows)
    """
    print("Building final targeting table...")
    
    # Create the final targeting table
    final_targeting, debug_rows = build_targeting(df_ads_enhanced, df_prod_group)

    return final_targeting, debug_rows


def main(start_date: str = "2025-10-01", end_date: str = "2025-12-31", cutoff_date: str = "2025-11-01", website_id: str = '6839260124a2adf314674a5e'):
    """
    Main function to execute the product matching workflow.

    Args:
        start_date: Start date for data retrieval in format 'YYYY-MM-DD'
        end_date: End date for data retrieval in format 'YYYY-MM-DD'
        cutoff_date: Cutoff date for metrics coalescing in format 'YYYY-MM-DD'
        website_id: Website ID to process
    """
    # Setup logging
    config = get_config()
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format')
    enable_file_logging = config.get('logging.enable_file_logging', False)
    log_file = config.get('logging.log_file', 'product_matching.log')

    setup_logging(
        level=log_level,
        log_format=log_format,
        enable_file_logging=enable_file_logging,
        log_file=log_file
    )

    import logging
    logging.info(f"Starting Product Matching Workflow for website {website_id} with date range: {start_date} to {end_date}, cutoff: {cutoff_date}")

    # Get system info for optimal worker configuration
    worker_info = get_system_workers_info()
    MAX_WORKERS = max(1, worker_info["recommended_io_bound"] - 1)
    logging.info(f"Using MAX_WORKERS = {MAX_WORKERS}")

    # Check if parquet files exist in the parent directory
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_files = {
        'websites': os.path.join(parent_dir, 'websites.parquet'),
        'df_ads_ori': os.path.join(parent_dir, 'df_ads_ori.parquet'),
        'df_prod': os.path.join(parent_dir, 'df_prod.parquet'),
        'df_prod_group': os.path.join(parent_dir, 'df_prod_group.parquet'),
        'df_metrics': os.path.join(parent_dir, 'df_metrics.parquet'),
        'df_metrics_by_country': os.path.join(parent_dir, 'df_metrics_by_country.parquet'),
        'df_metrics_by_products': os.path.join(parent_dir, 'df_metrics_by_products.parquet')
    }

    # Check if all parquet files exist
    all_exist = all(os.path.exists(file) for file in parquet_files.values())

    if not all_exist:
        logging.error("Required parquet files not found in the parent directory.")
        logging.error("Expected files:")
        for name, path in parquet_files.items():
            exists = "✓" if os.path.exists(path) else "✗"
            logging.error(f"  {exists} {path}")
        logging.error("Either run the original notebook to generate these files, or ensure they exist.")
        logging.info("Note: The date and website ID parameters are only used when fetching fresh data from ClickHouse.")
        return

    logging.info(f"Loading data from existing parquet files for website {website_id}...")
    websites = pd.read_parquet(parquet_files['websites'])
    df_ads_ori = pd.read_parquet(parquet_files['df_ads_ori'])
    df_prod = pd.read_parquet(parquet_files['df_prod'])
    df_prod_group = pd.read_parquet(parquet_files['df_prod_group'])
    df_metrics = pd.read_parquet(parquet_files['df_metrics'])
    df_metrics_by_country = pd.read_parquet(parquet_files['df_metrics_by_country'])
    df_metrics_by_products = pd.read_parquet(parquet_files['df_metrics_by_products'])

    # Filter data by website ID if needed
    logging.info(f"Filtering data for website ID: {website_id}")
    df_ads_ori = df_ads_ori[df_ads_ori['websiteId'] == website_id] if 'websiteId' in df_ads_ori.columns else df_ads_ori
    df_prod = df_prod[df_prod['websiteId'] == website_id] if 'websiteId' in df_prod.columns else df_prod
    df_prod_group = df_prod_group[df_prod_group['websiteId'] == website_id] if 'websiteId' in df_prod_group.columns else df_prod_group
    df_metrics = df_metrics[df_metrics['websiteId'] == website_id] if 'websiteId' in df_metrics.columns else df_metrics
    df_metrics_by_country = df_metrics_by_country[df_metrics_by_country['websiteId'] == website_id] if 'websiteId' in df_metrics_by_country.columns else df_metrics_by_country
    df_metrics_by_products = df_metrics_by_products[df_metrics_by_products['websiteId'] == website_id] if 'websiteId' in df_metrics_by_products.columns else df_metrics_by_products

    logging.info(f"Loaded filtered datasets for website {website_id}:")
    logging.info(f"  - websites: {websites.shape}")
    logging.info(f"  - df_ads_ori: {df_ads_ori.shape}")
    logging.info(f"  - df_prod: {df_prod.shape}")
    logging.info(f"  - df_prod_group: {df_prod_group.shape}")
    logging.info(f"  - df_metrics: {df_metrics.shape}")
    logging.info(f"  - df_metrics_by_country: {df_metrics_by_country.shape}")
    logging.info(f"  - df_metrics_by_products: {df_metrics_by_products.shape}")

    # Process destination URLs
    df_ads_exploded, df_ads_unique_urls, df_ads_exploded_dist = process_urls(df_ads_ori, MAX_WORKERS)

    # Process product groups
    df_prod_subset_exploded = process_product_groups(df_prod_group)

    # Coalesce metrics with specified cutoff date
    logging.info(f"Applying cutoff date {cutoff_date} for metrics coalescing...")
    coalesced_df, temp_df = coalesce_metrics(df_metrics_by_products, df_metrics_by_country, cutoff_date)

    # Enhance ads data
    df_ads_enhanced, df_prod_group_enhanced = enhance_ads_data(
        df_ads_ori, df_ads_exploded_dist, df_prod_group, MAX_WORKERS
    )

    # Classify product IDs
    coalesced_df = classify_product_ids(coalesced_df, df_prod, MAX_WORKERS)

    # Build final targeting table
    final_targeting, debug_rows = build_final_targeting(df_ads_enhanced, df_prod_group_enhanced)

    # Identify main products by merging final_targeting with coalesced_df
    logging.info("Identifying main products by matching campaign IDs and product group IDs...")
    coalesced_df_with_flags = identify_main_products_vectorized(final_targeting, coalesced_df)

    # Print results
    logging.info(f"Generated result tables for website {website_id}:")
    logging.info(f"  - final_targeting: {final_targeting.shape}")
    logging.info(f"  - coalesced_df: {coalesced_df.shape}")
    logging.info(f"  - coalesced_df with main product flags: {coalesced_df_with_flags.shape}")
    logging.info(f"  - df_ads_exploded: {df_ads_exploded.shape}")
    logging.info(f"  - debug_rows: {debug_rows.shape}")

    # Show sample of final_targeting (the main result)
    logging.info(f"Sample of final_targeting table:")
    logging.info(f"{final_targeting.head()}")

    # Show sample of coalesced_df with flags
    logging.info(f"Sample of coalesced_df with main_product_flag:")
    logging.info(f"{coalesced_df_with_flags.head()}")

    # Count how many rows have the main product flag set to 1
    flagged_count = coalesced_df_with_flags[coalesced_df_with_flags['main_product_flag'] == 1].shape[0]
    logging.info(f"Number of rows with main_product_flag = 1: {flagged_count}")

    # Save results to parquet files
    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    final_targeting.to_parquet(os.path.join(results_dir, 'final_targeting.parquet'))
    coalesced_df.to_parquet(os.path.join(results_dir, 'coalesced_df.parquet'))
    coalesced_df_with_flags.to_parquet(os.path.join(results_dir, 'coalesced_df_with_flags.parquet'))
    df_ads_exploded.to_parquet(os.path.join(results_dir, 'df_ads_exploded.parquet'))
    debug_rows.to_parquet(os.path.join(results_dir, 'debug_rows.parquet'))

    logging.info(f"Results saved to {results_dir}/")
    logging.info("Files created:")
    logging.info("  - final_targeting.parquet (main result)")
    logging.info("  - coalesced_df.parquet")
    logging.info("  - coalesced_df_with_flags.parquet (with main product flags)")
    logging.info("  - df_ads_exploded.parquet")
    logging.info("  - debug_rows.parquet")

    logging.info("Workflow completed successfully!")


def run_with_clickhouse_connection(start_date: str = "2025-10-01", end_date: str = "2025-12-31", cutoff_date: str = "2025-11-01", website_id: str = '6839260124a2adf314674a5e'):
    """
    Alternative entry point that connects to ClickHouse directly instead of using parquet files.

    Args:
        start_date: Start date for data retrieval in format 'YYYY-MM-DD'
        end_date: End date for data retrieval in format 'YYYY-MM-DD'
        cutoff_date: Cutoff date for metrics coalescing in format 'YYYY-MM-DD'
        website_id: Website ID to process
    """
    # Setup logging
    config = get_config()
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format')
    enable_file_logging = config.get('logging.enable_file_logging', False)
    log_file = config.get('logging.log_file', 'product_matching.log')

    setup_logging(
        level=log_level,
        log_format=log_format,
        enable_file_logging=enable_file_logging,
        log_file=log_file
    )

    import logging
    logging.info(f"Starting Product Matching Workflow with ClickHouse connection for website {website_id}, date range: {start_date} to {end_date}, cutoff: {cutoff_date}")

    # Get system info for optimal worker configuration
    worker_info = get_system_workers_info()
    MAX_WORKERS = max(1, worker_info["recommended_io_bound"] - 1)
    logging.info(f"Using MAX_WORKERS = {MAX_WORKERS}")

    # Initialize credentials and create ClickHouse clients
    try:
        credentials = initialize_credentials(profile_name='live')
        client = create_clickhouse_client(credentials)
        client_staging = create_clickhouse_client_staging(credentials)
    except Exception as e:
        logging.error(f"Failed to establish ClickHouse connection: {e}")
        return

    # Load or fetch data
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    websites, df_ads_ori, df_prod, df_prod_group, df_metrics, df_metrics_by_country, df_metrics_by_products = \
        load_or_fetch_data(client, wid=website_id, start_date=start_date, end_date=end_date, parquet_dir=parent_dir)

    logging.info(f"Loaded datasets for website {website_id}:")
    logging.info(f"  - websites: {websites.shape}")
    logging.info(f"  - df_ads_ori: {df_ads_ori.shape}")
    logging.info(f"  - df_prod: {df_prod.shape}")
    logging.info(f"  - df_prod_group: {df_prod_group.shape}")
    logging.info(f"  - df_metrics: {df_metrics.shape}")
    logging.info(f"  - df_metrics_by_country: {df_metrics_by_country.shape}")
    logging.info(f"  - df_metrics_by_products: {df_metrics_by_products.shape}")

    # Process destination URLs
    df_ads_exploded, df_ads_unique_urls, df_ads_exploded_dist = process_urls(df_ads_ori, MAX_WORKERS)

    # Process product groups
    df_prod_subset_exploded = process_product_groups(df_prod_group)

    # Coalesce metrics with specified cutoff date
    logging.info(f"Applying cutoff date {cutoff_date} for metrics coalescing...")
    coalesced_df, temp_df = coalesce_metrics(df_metrics_by_products, df_metrics_by_country, cutoff_date)

    # Enhance ads data
    df_ads_enhanced, df_prod_group_enhanced = enhance_ads_data(
        df_ads_ori, df_ads_exploded_dist, df_prod_group, MAX_WORKERS
    )

    # Classify product IDs
    coalesced_df = classify_product_ids(coalesced_df, df_prod, MAX_WORKERS)

    # Build final targeting table
    final_targeting, debug_rows = build_final_targeting(df_ads_enhanced, df_prod_group_enhanced)

    # Identify main products by merging final_targeting with coalesced_df
    logging.info("Identifying main products by matching campaign IDs and product group IDs...")
    coalesced_df_with_flags = identify_main_products_vectorized(final_targeting, coalesced_df)

    # Print results
    logging.info(f"Generated result tables for website {website_id}:")
    logging.info(f"  - final_targeting: {final_targeting.shape}")
    logging.info(f"  - coalesced_df: {coalesced_df.shape}")
    logging.info(f"  - coalesced_df with main product flags: {coalesced_df_with_flags.shape}")
    logging.info(f"  - df_ads_exploded: {df_ads_exploded.shape}")
    logging.info(f"  - debug_rows: {debug_rows.shape}")

    # Show sample of final_targeting (the main result)
    logging.info(f"Sample of final_targeting table:")
    logging.info(f"{final_targeting.head()}")

    # Show sample of coalesced_df with flags
    logging.info(f"Sample of coalesced_df with main_product_flag:")
    logging.info(f"{coalesced_df_with_flags.head()}")

    # Count how many rows have the main product flag set to 1
    flagged_count = coalesced_df_with_flags[coalesced_df_with_flags['main_product_flag'] == 1].shape[0]
    logging.info(f"Number of rows with main_product_flag = 1: {flagged_count}")

    # Save results to parquet files
    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    final_targeting.to_parquet(os.path.join(results_dir, 'final_targeting.parquet'))
    coalesced_df.to_parquet(os.path.join(results_dir, 'coalesced_df.parquet'))
    coalesced_df_with_flags.to_parquet(os.path.join(results_dir, 'coalesced_df_with_flags.parquet'))
    df_ads_exploded.to_parquet(os.path.join(results_dir, 'df_ads_exploded.parquet'))
    debug_rows.to_parquet(os.path.join(results_dir, 'debug_rows.parquet'))

    logging.info(f"Results saved to {results_dir}/")
    logging.info("Files created:")
    logging.info("  - final_targeting.parquet (main result)")
    logging.info("  - coalesced_df.parquet")
    logging.info("  - coalesced_df_with_flags.parquet (with main product flags)")
    logging.info("  - df_ads_exploded.parquet")
    logging.info("  - debug_rows.parquet")

    logging.info("Workflow completed successfully!")


if __name__ == "__main__":
    # By default, run with existing parquet files
    # To run with ClickHouse connection instead, uncomment the next line:
    # run_with_clickhouse_connection(start_date="2025-10-01", end_date="2025-12-31", cutoff_date="2025-11-01", website_id='6839260124a2adf314674a5e')

    main(start_date="2025-10-01", end_date="2025-12-31", cutoff_date="2025-11-01", website_id='6839260124a2adf314674a5e')
