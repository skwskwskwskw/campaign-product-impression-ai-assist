"""Compute campaign × product metrics from a coalesced product table."""

from __future__ import annotations

import argparse
from typing import Dict, Optional

from metric import compute_metrics, load_dataframe, write_outputs


def _build_overrides(args: argparse.Namespace) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if args.date_col:
        overrides["date"] = args.date_col
    if args.platform_col:
        overrides["platform"] = args.platform_col
    if args.campaign_id_col:
        overrides["campaignId"] = args.campaign_id_col
    if args.campaign_name_col:
        overrides["campaignName"] = args.campaign_name_col
    if args.adset_id_col:
        overrides["adSetId"] = args.adset_id_col
    if args.adset_name_col:
        overrides["adSetName"] = args.adset_name_col
    if args.ad_id_col:
        overrides["adId"] = args.ad_id_col
    if args.ad_name_col:
        overrides["adName"] = args.ad_name_col
    if args.product_id_col:
        overrides["productId"] = args.product_id_col
    if args.product_group_id_col:
        overrides["productGroupId"] = args.product_group_id_col
    if args.product_name_col:
        overrides["productName"] = args.product_name_col
    if args.product_group_name_col:
        overrides["productGroupName"] = args.product_group_name_col
    if args.lead_col:
        overrides["isLead"] = args.lead_col
    if args.sku_weight_col:
        overrides["sku_weight"] = args.sku_weight_col

    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ad-level, SKU allocation, and SKU performance metrics "
            "from a coalesced product table."
        )
    )
    parser.add_argument("--input", required=True, help="Path to coalesced_df CSV or Parquet file")
    parser.add_argument("--output-dir", default="metric-output", help="Directory to store output files")
    parser.add_argument(
        "--profit-col",
        default="conversionsValue",
        help="Column to treat as gross profit / revenue",
    )
    parser.add_argument("--weight-col", dest="weight_col", help="Column to use for lead-only weights")

    parser.add_argument("--date-col")
    parser.add_argument("--platform-col")
    parser.add_argument("--campaign-id-col")
    parser.add_argument("--campaign-name-col")
    parser.add_argument("--adset-id-col")
    parser.add_argument("--adset-name-col")
    parser.add_argument("--ad-id-col")
    parser.add_argument("--ad-name-col")
    parser.add_argument("--product-id-col")
    parser.add_argument("--product-group-id-col")
    parser.add_argument("--product-name-col")
    parser.add_argument("--product-group-name-col")
    parser.add_argument("--lead-col")
    parser.add_argument("--sku-weight-col")

    parser.add_argument(
        "--formats",
        default="csv",
        help="Comma-separated list of output formats (csv, parquet)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = load_dataframe(args.input)
    overrides = _build_overrides(args)

    result = compute_metrics(
        df,
        profit_col=args.profit_col,
        lead_col=args.lead_col,
        weight_col=args.weight_col,
        column_overrides=overrides or None,
    )

    outputs = {
        "ad_data": result.ad_data,
        "sku_allocation": result.sku_allocation,
        "sku_performance": result.sku_performance,
    }

    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    write_outputs(outputs, args.output_dir, formats=formats)


if __name__ == "__main__":
    main()
