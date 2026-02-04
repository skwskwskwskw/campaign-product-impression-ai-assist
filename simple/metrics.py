"""
Simplified Metrics Computation for Ad-Product Attribution.

This module computes:
1. Fair allocation: Distributes ad spend/impressions proportionally to all attributed products
2. Lead-only allocation: Distributes only to lead products (directly targeted)

Output tables:
- ad_data: Ad-level totals (spend, impressions, gross profit)
- sku_allocation: Product-level with both fair and lead-only metrics
- sku_performance: Aggregated product performance across all ads

Usage:
    from simple.metrics import compute_all_metrics

    results = compute_all_metrics(
        df=metrics_df,           # DataFrame with isLead flag
        profit_col="grossProfit",
        weight_col="conversions"
    )
    results.ad_data          # Ad totals
    results.sku_allocation   # Per-product allocation
    results.sku_performance  # Product summary
"""

from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import numpy as np


@dataclass
class MetricsResult:
    """Container for computed metrics."""
    ad_data: pd.DataFrame
    sku_allocation: pd.DataFrame
    sku_performance: pd.DataFrame


# ============================================================
# Helper Functions
# ============================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric, filling NaN with 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def safe_sum(series: pd.Series) -> float:
    """Sum series safely."""
    return float(safe_numeric(series).sum())


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely, returning 0 where denominator is 0."""
    denom = denominator.replace({0: np.nan})
    return (numerator / denom).fillna(0.0)


def normalize_lead_flag(series: pd.Series) -> pd.Series:
    """Normalize isLead column to 0/1 integers."""
    if series.dtype == bool:
        return series.astype(int)

    s = series.astype(str).str.strip().str.lower()
    mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}
    return s.map(mapping).fillna(0).astype(int)


# ============================================================
# Main Computation Function
# ============================================================

def compute_all_metrics(
    df: pd.DataFrame,
    profit_col: str = "grossProfit",
    weight_col: Optional[str] = None,
    lead_col: str = "isLead",
) -> MetricsResult:
    """
    Compute fair and lead-only metrics allocation.

    How allocation works:
    1. Fair allocation: Each product gets a share proportional to its weight
       - share = product_weight / total_weight_for_ad
       - spend_fair = ad_spend * share

    2. Lead-only allocation: Same as fair, but only among lead products
       - share_lead = lead_weight / total_lead_weight_for_ad
       - spend_lead_only = ad_spend * share_lead

    Args:
        df: Input DataFrame with columns:
            - date, platform, campaignId, adSetId, adId
            - productId (or productGroupId)
            - spend, impressions, clicks (metrics)
            - isLead (or main_product_flag)
            - A profit column (default: grossProfit)
            - A weight column (default: auto-detect)

        profit_col: Column name for gross profit/revenue
        weight_col: Column for allocation weights (auto-detects if None)
        lead_col: Column indicating lead products (1) vs halo (0)

    Returns:
        MetricsResult with ad_data, sku_allocation, sku_performance
    """
    if df.empty:
        empty = pd.DataFrame()
        return MetricsResult(ad_data=empty, sku_allocation=empty, sku_performance=empty)

    data = df.copy()

    # === Normalize columns ===

    # Date
    if "date" not in data.columns and "timestamp" in data.columns:
        data["date"] = pd.to_datetime(data["timestamp"], errors="coerce").dt.date
    elif "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date

    # Product ID
    if "productId" not in data.columns:
        if "productGroupId" in data.columns:
            data["productId"] = data["productGroupId"]
        elif "productGroupId_" in data.columns:
            data["productId"] = data["productGroupId_"]
        else:
            raise ValueError("No product ID column found (productId, productGroupId)")

    # String columns
    str_cols = ["platform", "campaignId", "adSetId", "adId", "productId"]
    for col in str_cols:
        if col in data.columns:
            data[col] = data[col].fillna("").astype(str)

    # Name columns (optional)
    name_cols = ["campaignName", "adSetName", "adName", "productGroupId", "productGroupName", "productName"]
    for col in name_cols:
        if col not in data.columns:
            data[col] = ""
        else:
            data[col] = data[col].fillna("").astype(str)

    # Metric columns
    metric_cols = ["spend", "impressions", "clicks", "interactions", "conversions"]
    for col in metric_cols:
        if col in data.columns:
            data[col] = safe_numeric(data[col])
        else:
            data[col] = 0.0

    # Gross profit
    if profit_col in data.columns:
        data["grossProfit"] = safe_numeric(data[profit_col])
    elif "grossProfit" not in data.columns:
        data["grossProfit"] = 0.0
    else:
        data["grossProfit"] = safe_numeric(data["grossProfit"])

    # Lead flag
    if lead_col in data.columns:
        data["isLead"] = normalize_lead_flag(data[lead_col])
    elif "main_product_flag" in data.columns:
        data["isLead"] = normalize_lead_flag(data["main_product_flag"])
    else:
        data["isLead"] = 0

    # Weight column (for allocation)
    if weight_col and weight_col in data.columns:
        data["sku_weight"] = safe_numeric(data[weight_col])
    else:
        # Auto-detect: prefer conversions > clicks > impressions > spend > 1
        for candidate in ["conversions", "clicks", "impressions", "spend"]:
            if candidate in data.columns and data[candidate].sum() > 0:
                data["sku_weight"] = safe_numeric(data[candidate])
                break
        else:
            data["sku_weight"] = 1.0

    # Match stage column (for tracking confidence of product group match)
    if "match_stage" not in data.columns:
        data["match_stage"] = "unmatched"
    else:
        data["match_stage"] = data["match_stage"].fillna("unmatched").astype(str)

    # === Aggregate to (date, ad, product) level ===
    dims = [
        "date", "productId", "productGroupId", "productGroupName", "productName",
        "platform", "campaignId", "campaignName", "adSetId", "adSetName",
        "adId", "adName", "isLead", "match_stage"
    ]
    dims = [d for d in dims if d in data.columns]

    agg_cols = ["spend", "impressions", "clicks", "grossProfit", "sku_weight"]
    sku = data.groupby(dims, dropna=False)[agg_cols].sum().reset_index()

    # === Compute ad-level totals ===
    ad_keys = ["date", "platform", "campaignId", "adSetId", "adId"]
    ad_keys = [k for k in ad_keys if k in sku.columns]

    ad_totals = sku.groupby(ad_keys, dropna=False).agg(
        ad_spend_total=("spend", safe_sum),
        ad_impressions_total=("impressions", safe_sum),
        ad_clicks_total=("clicks", safe_sum),
        ad_gross_profit_total=("grossProfit", safe_sum),
        total_weight_all=("sku_weight", safe_sum),
    ).reset_index()

    sku = sku.merge(ad_totals, on=ad_keys, how="left")

    # === Compute fair share (all products) ===
    sku["share"] = safe_div(sku["sku_weight"], sku["total_weight_all"])

    # === Compute lead-only share ===
    sku["lead_weight"] = np.where(sku["isLead"] == 1, sku["sku_weight"], 0.0)
    sku["total_weight_lead"] = sku.groupby(ad_keys, dropna=False)["lead_weight"].transform("sum")
    sku["share_lead_only"] = safe_div(sku["lead_weight"], sku["total_weight_lead"])

    # === Compute allocated metrics ===
    # Fair allocation (proportional to all products)
    sku["spend_fair"] = sku["spend"]  # Already at product level
    sku["impressions_fair"] = sku["impressions"]
    sku["clicks_fair"] = sku["clicks"]
    sku["gross_profit_fair"] = sku["grossProfit"]

    # Lead-only allocation (full ad total distributed only among leads)
    sku["spend_lead_only"] = sku["ad_spend_total"] * sku["share_lead_only"]
    sku["impressions_lead_only"] = sku["ad_impressions_total"] * sku["share_lead_only"]
    sku["clicks_lead_only"] = sku["ad_clicks_total"] * sku["share_lead_only"]
    sku["gross_profit_lead_only"] = sku["ad_gross_profit_total"] * sku["share_lead_only"]

    # === Build output tables ===

    # SKU Allocation table
    sku_alloc_cols = [
        "date", "productId", "productGroupId", "productGroupName", "productName",
        "platform", "campaignId", "campaignName", "adSetId", "adSetName",
        "adId", "adName", "isLead", "match_stage", "sku_weight", "share",
        "gross_profit_fair", "spend_fair", "impressions_fair", "clicks_fair",
        "gross_profit_lead_only", "spend_lead_only", "impressions_lead_only", "clicks_lead_only",
    ]
    sku_alloc_cols = [c for c in sku_alloc_cols if c in sku.columns]
    sku_allocation = sku[sku_alloc_cols].copy()

    # Ad Data table (unique ads with totals)
    ad_data_cols = [
        "date", "platform", "campaignId", "campaignName",
        "adSetId", "adSetName", "adId", "adName",
        "ad_spend_total", "ad_impressions_total", "ad_clicks_total", "ad_gross_profit_total",
    ]
    ad_data_cols = [c for c in ad_data_cols if c in sku.columns]
    ad_data = sku[ad_data_cols].drop_duplicates(ad_keys)

    # SKU Performance table (product summary across all ads)
    sku_perf = sku_allocation.groupby(["date", "productId"], dropna=False).agg(
        sku_spend_fair=("spend_fair", safe_sum),
        sku_impressions_fair=("impressions_fair", safe_sum),
        sku_clicks_fair=("clicks_fair", safe_sum),
        sku_gross_profit_fair=("gross_profit_fair", safe_sum),
        sku_spend_lead_only=("spend_lead_only", safe_sum),
        sku_impressions_lead_only=("impressions_lead_only", safe_sum),
        sku_clicks_lead_only=("clicks_lead_only", safe_sum),
        sku_gross_profit_lead_only=("gross_profit_lead_only", safe_sum),
    ).reset_index()

    # Add product dimensions
    product_dims = ["productId", "productGroupId", "productGroupName", "productName"]
    product_dims = [c for c in product_dims if c in sku_allocation.columns]
    if product_dims:
        product_info = sku_allocation[product_dims].drop_duplicates("productId")
        sku_perf = sku_perf.merge(product_info, on="productId", how="left")

    return MetricsResult(
        ad_data=ad_data,
        sku_allocation=sku_allocation,
        sku_performance=sku_perf,
    )


def compute_summary_kpis(sku_allocation: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary KPIs from SKU allocation data.

    This is useful for quick analysis and dashboards.

    Args:
        sku_allocation: Output from compute_all_metrics

    Returns:
        DataFrame with summary KPIs per campaign
    """
    if sku_allocation.empty:
        return pd.DataFrame()

    dims = ["platform", "campaignId", "campaignName"]
    dims = [d for d in dims if d in sku_allocation.columns]

    summary = sku_allocation.groupby(dims, dropna=False).agg(
        gp_total=("gross_profit_fair", "sum"),
        spend_total=("spend_fair", "sum"),
        impr_total=("impressions_fair", "sum"),
        gp_lead=("gross_profit_lead_only", "sum"),
        spend_lead=("spend_lead_only", "sum"),
        impr_lead=("impressions_lead_only", "sum"),
    ).reset_index()

    # Compute halo (spillover) metrics
    summary["gp_halo"] = summary["gp_total"] - summary["gp_lead"]
    summary["spend_halo"] = summary["spend_total"] - summary["spend_lead"]
    summary["impr_halo"] = summary["impr_total"] - summary["impr_lead"]

    # Efficiency metrics
    summary["gp_per_spend_total"] = safe_div(summary["gp_total"], summary["spend_total"])
    summary["gp_per_spend_lead"] = safe_div(summary["gp_lead"], summary["spend_lead"])
    summary["gp_per_1k_impr_total"] = 1000 * safe_div(summary["gp_total"], summary["impr_total"])
    summary["gp_per_1k_impr_lead"] = 1000 * safe_div(summary["gp_lead"], summary["impr_lead"])

    # Spillover ratio
    summary["spillover_share"] = safe_div(summary["gp_halo"], summary["gp_total"])

    return summary


def write_outputs(
    result: MetricsResult,
    output_dir: str,
    formats: List[str] = None,
    prefix: str = "",
) -> None:
    """
    Write metrics results to files.

    Args:
        result: MetricsResult from compute_all_metrics
        output_dir: Directory to write files
        formats: List of formats ("csv", "parquet")
        prefix: Optional prefix for filenames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if formats is None:
        formats = ["csv", "parquet"]

    tables = {
        "ad_data": result.ad_data,
        "sku_allocation": result.sku_allocation,
        "sku_performance": result.sku_performance,
    }

    for name, df in tables.items():
        if df.empty:
            continue

        filename = f"{prefix}{name}" if prefix else name

        if "csv" in formats:
            df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)

        if "parquet" in formats:
            df.to_parquet(os.path.join(output_dir, f"{filename}.parquet"), index=False)
