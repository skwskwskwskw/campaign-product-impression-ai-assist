#!/usr/bin/env python3
"""
Unified Pipeline Runner for Campaign Product Impression Analysis

This script runs the complete pipeline from data fetch to dashboard-ready output.

WORKFLOW:
=========
Step 1: Fetch raw data from ClickHouse (or use existing parquet files)
Step 2: Run product matching to identify lead/halo products
Step 3: Compute metrics allocation
Step 4: Output files ready for Streamlit dashboard

USAGE:
======
# Option A: Fetch data from ClickHouse staging (no AWS needed)
python run_pipeline.py --from-clickhouse --use-staging

# Option B: Fetch data from ClickHouse production (requires AWS)
python run_pipeline.py --from-clickhouse --aws-profile live

# Option C: Use existing parquet files in repo root
python run_pipeline.py --from-parquet

# Option D: Use simple pipeline with data in ./data directory
python run_pipeline.py --simple --data-dir ./data

After running:
    streamlit run app.py
    # Upload results/metric-output/sku_allocation.csv
"""

import argparse
import os
import sys
import subprocess


def check_data_files(data_dir: str, file_names: list) -> dict:
    """Check which data files exist."""
    results = {}
    for name in file_names:
        path = os.path.join(data_dir, name)
        results[name] = os.path.exists(path)
    return results


def run_command(cmd: list, description: str) -> bool:
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
REQUIRED DATA FILES:
====================

For --from-parquet (full pipeline):
  - websites.parquet
  - df_ads_ori.parquet
  - df_prod.parquet
  - df_prod_group.parquet
  - df_metrics.parquet
  - df_metrics_by_country.parquet
  - df_metrics_by_products.parquet

For --simple (simple pipeline):
  - data/ads.parquet (or ads.csv)
  - data/products.parquet (or products.csv)
  - data/metrics.parquet (or metrics.csv)

OUTPUT:
=======
  results/metric-output/sku_allocation.csv  <- Use this in Streamlit
  results/metric-output/sku_allocation.parquet
  results/metric-output/ad_data.csv
  results/metric-output/sku_performance.csv
        """
    )

    # Data source options
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-parquet", action="store_true",
        help="Use existing parquet files in repo root (full pipeline)"
    )
    source.add_argument(
        "--from-clickhouse", action="store_true",
        help="Fetch data from ClickHouse database"
    )
    source.add_argument(
        "--simple", action="store_true",
        help="Run simple pipeline (requires ads/products/metrics files)"
    )

    # ClickHouse options
    parser.add_argument(
        "--use-staging", action="store_true",
        help="Use staging ClickHouse (no AWS needed)"
    )
    parser.add_argument(
        "--aws-profile", default="live",
        help="AWS profile for ClickHouse credentials"
    )

    # Common options
    parser.add_argument(
        "--website-id", default="6839260124a2adf314674a5e",
        help="Website ID to process"
    )
    parser.add_argument(
        "--start-date", default="2025-10-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2025-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--cutoff-date", default="2025-11-01",
        help="Cutoff date for metrics coalescing"
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="Data directory for simple pipeline"
    )
    parser.add_argument(
        "--output-dir", default="./results",
        help="Output directory"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CAMPAIGN PRODUCT IMPRESSION PIPELINE")
    print("=" * 60)

    # Check which mode
    if args.simple:
        # Simple pipeline
        required_files = ["ads.parquet", "products.parquet", "metrics.parquet"]
        alt_files = ["ads.csv", "products.csv", "metrics.csv"]

        found = check_data_files(args.data_dir, required_files + alt_files)

        has_parquet = all(found.get(f, False) for f in required_files)
        has_csv = all(found.get(f, False) for f in alt_files)

        if not has_parquet and not has_csv:
            print(f"\nERROR: Missing data files in {args.data_dir}")
            print("\nRequired files (parquet or csv):")
            for f in required_files:
                status = "FOUND" if found.get(f, False) else "MISSING"
                print(f"  {status}: {f}")
            for f in alt_files:
                status = "FOUND" if found.get(f, False) else "MISSING"
                print(f"  {status}: {f}")
            print(f"\nTo generate sample data for testing:")
            print(f"  python scripts/generate_sample_data.py --output-dir {args.data_dir}")
            sys.exit(1)

        cmd = [
            sys.executable, "-m", "simple.run_pipeline",
            "--from-parquet", args.data_dir,
            "--output-dir", args.output_dir,
            "--format", "both",
            "--verbose"
        ]

        if not run_command(cmd, "Simple Pipeline"):
            sys.exit(1)

    elif args.from_parquet:
        # Full pipeline with existing parquet files
        required_files = [
            "websites.parquet",
            "df_ads_ori.parquet",
            "df_prod.parquet",
            "df_prod_group.parquet",
            "df_metrics.parquet",
            "df_metrics_by_country.parquet",
            "df_metrics_by_products.parquet"
        ]

        found = check_data_files(".", required_files)
        missing = [f for f, exists in found.items() if not exists]

        if missing:
            print(f"\nERROR: Missing parquet files in repo root")
            print("\nRequired files:")
            for f in required_files:
                status = "FOUND" if found[f] else "MISSING"
                print(f"  {status}: {f}")
            print("\nOptions:")
            print("  1. Run with --from-clickhouse --use-staging to fetch data")
            print("  2. Export parquet files from the Jupyter notebook")
            print("  3. Run --simple with data in ./data directory")
            sys.exit(1)

        cmd = [
            sys.executable, "product-matching.py",
            "--from-parquet",
            "--website-id", args.website_id,
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--cutoff-date", args.cutoff_date
        ]

        if not run_command(cmd, "Product Matching Pipeline"):
            sys.exit(1)

    elif args.from_clickhouse:
        # Full pipeline fetching from ClickHouse
        cmd = [
            sys.executable, "product-matching.py",
            "--from-clickhouse",
            "--website-id", args.website_id,
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--cutoff-date", args.cutoff_date
        ]

        if args.use_staging:
            cmd.append("--use-staging")
        else:
            cmd.extend(["--aws-profile", args.aws_profile])

        if not run_command(cmd, "Product Matching Pipeline (ClickHouse)"):
            sys.exit(1)

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    output_dir = args.output_dir
    if not args.simple:
        output_dir = os.path.join(args.output_dir, "metric-output")

    sku_csv = os.path.join(output_dir, "sku_allocation.csv")
    sku_parquet = os.path.join(output_dir, "sku_allocation.parquet")

    print(f"\nOutput files:")
    if os.path.exists(sku_csv):
        print(f"  - {sku_csv}")
    if os.path.exists(sku_parquet):
        print(f"  - {sku_parquet}")

    print(f"\nNext steps:")
    print(f"  1. Run: streamlit run app.py")
    print(f"  2. Upload: {sku_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
