#!/usr/bin/env python3
"""
Ad-Product Attribution Pipeline - Main Entry Point

This script identifies which products each ad is targeting and computes
metrics allocation (fair vs lead-only).

Usage:
    # From parquet files:
    python -m simple.run_pipeline --from-parquet ./data

    # From ClickHouse:
    python -m simple.run_pipeline --website-id 6839260124a2adf314674a5e \\
        --start-date 2025-10-01 --end-date 2025-12-31

    # With custom config:
    python -m simple.run_pipeline --fuzzy-threshold 90 --output-dir ./results

Output:
    - results/targeting.parquet: Ad -> Product mappings
    - results/ad_data.parquet: Ad-level metrics
    - results/sku_allocation.parquet: Product-level metrics with lead/halo
    - results/sku_performance.parquet: Product summary

For API usage, see simple.pipeline module.
"""

import argparse
import logging
import os
import sys
from typing import Optional, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple.config import PipelineConfig
from simple.ad_product_matcher import AdProductMatcher
from simple.metrics import compute_all_metrics, write_outputs, compute_summary_kpis


# ============================================================
# Data Loading
# ============================================================

def load_from_parquet(
    data_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from parquet files.

    Expected files:
        - df_ads_ori.parquet (or ads.parquet)
        - df_prod_group.parquet (or products.parquet)
        - df_metrics_by_products.parquet (or metrics.parquet)

    Returns:
        Tuple of (ads_df, products_df, metrics_df)
    """
    logging.info(f"Loading data from parquet files in {data_dir}")

    # Ads
    ads_files = ["df_ads_ori.parquet", "ads.parquet"]
    ads_df = None
    for f in ads_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            ads_df = pd.read_parquet(path)
            logging.info(f"Loaded ads: {ads_df.shape} from {f}")
            break

    if ads_df is None:
        raise FileNotFoundError(f"No ads file found in {data_dir}. Expected: {ads_files}")

    # Products
    prod_files = ["df_prod_group.parquet", "products.parquet", "df_prod.parquet"]
    products_df = None
    for f in prod_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            products_df = pd.read_parquet(path)
            logging.info(f"Loaded products: {products_df.shape} from {f}")
            break

    if products_df is None:
        raise FileNotFoundError(f"No products file found in {data_dir}. Expected: {prod_files}")

    # Metrics
    metrics_files = ["df_metrics_by_products.parquet", "metrics.parquet", "coalesced_df.parquet"]
    metrics_df = None
    for f in metrics_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            metrics_df = pd.read_parquet(path)
            logging.info(f"Loaded metrics: {metrics_df.shape} from {f}")
            break

    if metrics_df is None:
        logging.warning(f"No metrics file found in {data_dir}. Will skip metrics computation.")
        metrics_df = pd.DataFrame()

    return ads_df, products_df, metrics_df


def load_from_clickhouse(
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data directly from ClickHouse.

    Requires AWS credentials or local ClickHouse connection.

    Returns:
        Tuple of (ads_df, products_df, metrics_df)
    """
    logging.info(f"Loading data from ClickHouse for website {config.website_id}")

    # Import ClickHouse utilities from parent module
    try:
        from product.clickhouse_utils import (
            initialize_credentials,
            create_clickhouse_client,
        )
        from product.data_retrieval import (
            get_latest_ads,
            get_latest_product_groups,
            get_metrics_by_product,
        )
    except ImportError as e:
        raise ImportError(
            f"Could not import ClickHouse utilities: {e}. "
            "Make sure the 'product' module is available or use --from-parquet instead."
        )

    # Initialize connection
    if config.use_aws_secrets:
        credentials = initialize_credentials(profile_name=config.aws_profile)
        client = create_clickhouse_client(credentials)
    else:
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host=config.clickhouse_host,
            port=config.clickhouse_port,
            username=config.clickhouse_user,
            password=config.clickhouse_password,
            database=config.clickhouse_database,
        )

    # Fetch data
    ads_df = get_latest_ads(client, config.website_id)
    logging.info(f"Fetched ads: {ads_df.shape}")

    products_df = get_latest_product_groups(client, config.website_id)
    logging.info(f"Fetched products: {products_df.shape}")

    metrics_df = get_metrics_by_product(
        client, config.website_id, config.start_date, config.end_date
    )
    logging.info(f"Fetched metrics: {metrics_df.shape}")

    return ads_df, products_df, metrics_df


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(
    ads_df: pd.DataFrame,
    products_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config: PipelineConfig,
) -> dict:
    """
    Run the full ad-product attribution pipeline.

    Args:
        ads_df: Ads with destination URLs
        products_df: Product catalog
        metrics_df: Product-level metrics
        config: Pipeline configuration

    Returns:
        Dictionary with result DataFrames
    """
    logging.info("Starting ad-product attribution pipeline")

    # === Step 1: Match ads to products ===
    matcher = AdProductMatcher(config)
    matcher.build_product_index(products_df)

    # Normalize ads data
    ads = ads_df.copy()

    # Handle different column names
    if "destinationUrl" not in ads.columns:
        url_cols = ["destination_url", "url", "finalUrl"]
        for col in url_cols:
            if col in ads.columns:
                ads["destinationUrl"] = ads[col]
                break

    if "name" not in ads.columns:
        name_cols = ["adName", "ad_name"]
        for col in name_cols:
            if col in ads.columns:
                ads["name"] = ads[col]
                break

    # Build targeting table
    targeting_df = matcher.build_targeting_table(ads)
    logging.info(f"Built targeting table: {len(targeting_df)} ads")

    # === Step 2: Flag lead products in metrics ===
    metrics_flagged = pd.DataFrame()
    if not metrics_df.empty:
        # Determine product ID column
        pid_col = None
        for col in ["productGroupId", "productGroupId_", "productId"]:
            if col in metrics_df.columns:
                pid_col = col
                break

        if pid_col:
            metrics_flagged = matcher.flag_lead_products(
                targeting_df, metrics_df, product_id_col=pid_col
            )
            logging.info(f"Flagged metrics: {metrics_flagged['isLead'].sum()} lead products")
        else:
            logging.warning("No product ID column found in metrics, skipping lead flagging")
            metrics_flagged = metrics_df.copy()
            metrics_flagged["isLead"] = 0

    # === Step 3: Compute allocated metrics ===
    metrics_result = None
    if not metrics_flagged.empty:
        metrics_result = compute_all_metrics(
            metrics_flagged,
            profit_col=config.profit_column,
            weight_col=config.weight_column,
        )
        logging.info(
            f"Computed metrics: {len(metrics_result.ad_data)} ads, "
            f"{len(metrics_result.sku_allocation)} allocations, "
            f"{len(metrics_result.sku_performance)} products"
        )

    # === Step 4: Compute summary KPIs ===
    summary_df = pd.DataFrame()
    if metrics_result and not metrics_result.sku_allocation.empty:
        summary_df = compute_summary_kpis(metrics_result.sku_allocation)
        logging.info(f"Computed summary: {len(summary_df)} campaigns")

    return {
        "targeting": targeting_df,
        "ad_data": metrics_result.ad_data if metrics_result else pd.DataFrame(),
        "sku_allocation": metrics_result.sku_allocation if metrics_result else pd.DataFrame(),
        "sku_performance": metrics_result.sku_performance if metrics_result else pd.DataFrame(),
        "campaign_summary": summary_df,
    }


def save_results(results: dict, output_dir: str, formats: list) -> None:
    """Save all result tables to files."""
    os.makedirs(output_dir, exist_ok=True)

    for name, df in results.items():
        if df.empty:
            continue

        if "csv" in formats:
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

        if "parquet" in formats:
            df.to_parquet(os.path.join(output_dir, f"{name}.parquet"), index=False)

        logging.info(f"Saved {name}: {df.shape}")

    logging.info(f"All results saved to {output_dir}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ad-Product Attribution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from parquet files in current directory:
  python -m simple.run_pipeline --from-parquet .

  # Load from ClickHouse with custom date range:
  python -m simple.run_pipeline --website-id 6839260124a2adf314674a5e \\
      --start-date 2025-10-01 --end-date 2025-12-31

  # Use stricter fuzzy matching:
  python -m simple.run_pipeline --from-parquet . --fuzzy-threshold 90
        """
    )

    # Data source
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-parquet", metavar="DIR",
        help="Load data from parquet files in this directory"
    )
    source.add_argument(
        "--from-clickhouse", action="store_true",
        help="Load data from ClickHouse (requires credentials)"
    )

    # ClickHouse options
    parser.add_argument("--website-id", default="6839260124a2adf314674a5e",
                        help="Website ID to process")
    parser.add_argument("--start-date", default="2025-10-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--aws-profile", default="live",
                        help="AWS profile for credentials")

    # Matching options
    parser.add_argument("--fuzzy-threshold", type=int, default=85,
                        help="Fuzzy match threshold (0-100, higher=stricter)")
    parser.add_argument("--max-products", type=int, default=50,
                        help="Max products to match per ad")

    # Output options
    parser.add_argument("--output-dir", "-o", default="./results",
                        help="Output directory")
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="both",
                        help="Output format")

    # General options
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Build config
    config = PipelineConfig(
        website_id=args.website_id,
        start_date=args.start_date,
        end_date=args.end_date,
        aws_profile=args.aws_profile,
        fuzzy_threshold=args.fuzzy_threshold,
        max_products_per_ad=args.max_products,
        output_dir=args.output_dir,
    )

    # Determine output formats
    formats = []
    if args.format in ["csv", "both"]:
        formats.append("csv")
    if args.format in ["parquet", "both"]:
        formats.append("parquet")

    # Load data
    try:
        if args.from_parquet:
            ads_df, products_df, metrics_df = load_from_parquet(args.from_parquet)
        else:
            ads_df, products_df, metrics_df = load_from_clickhouse(config)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Run pipeline
    try:
        results = run_pipeline(ads_df, products_df, metrics_df, config)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

    # Save results
    save_results(results, args.output_dir, formats)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)

    if not results["targeting"].empty:
        matched = results["targeting"]["match_stage"] != "unmatched"
        print(f"\nAds processed: {len(results['targeting'])}")
        print(f"  - Matched: {matched.sum()} ({100*matched.mean():.1f}%)")

        stages = results["targeting"]["match_stage"].value_counts()
        for stage, count in stages.items():
            print(f"    - {stage}: {count}")

    if not results["sku_allocation"].empty:
        lead_count = results["sku_allocation"]["isLead"].sum()
        total_count = len(results["sku_allocation"])
        print(f"\nProduct allocations: {total_count}")
        print(f"  - Lead products: {lead_count} ({100*lead_count/total_count:.1f}%)")
        print(f"  - Halo products: {total_count - lead_count} ({100*(total_count-lead_count)/total_count:.1f}%)")

    if not results["campaign_summary"].empty:
        top = results["campaign_summary"].nlargest(5, "gp_total")
        print("\nTop 5 campaigns by gross profit:")
        for _, row in top.iterrows():
            print(
                f"  - {row.get('campaignName', row.get('campaignId', 'N/A'))[:40]}: "
                f"GP=${row['gp_total']:,.0f}, Spillover={row['spillover_share']*100:.1f}%"
            )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
