"""Metric computation pipeline for coalesced product data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import apply_column_map, ensure_date_column, infer_column_map


@dataclass
class MetricsResult:
    ad_data: pd.DataFrame
    sku_allocation: pd.DataFrame
    sku_performance: pd.DataFrame


def _select_weight_column(df: pd.DataFrame, weight_col: Optional[str]) -> pd.Series:
    if weight_col and weight_col in df.columns:
        return pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)

    for candidate in ["sku_weight", "quantity", "conversions", "clicks", "impressions", "spend"]:
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce").fillna(0.0)

    return pd.Series(1.0, index=df.index)


def _normalize_lead_flag(df: pd.DataFrame, lead_col: Optional[str]) -> pd.Series:
    if lead_col and lead_col in df.columns:
        series = df[lead_col]
    elif "isLead" in df.columns:
        series = df["isLead"]
    else:
        return pd.Series(0, index=df.index, dtype=int)

    if series.dtype == bool:
        return series.astype(int)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0})
        .fillna(0)
        .astype(int)
    )


def _sum_safe(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def compute_metrics(
    coalesced_df: pd.DataFrame,
    profit_col: str = "grossProfit",
    lead_col: Optional[str] = None,
    weight_col: Optional[str] = None,
    column_overrides: Optional[Dict[str, str]] = None,
) -> MetricsResult:
    """
    Compute ad-level, SKU allocation, and SKU performance metrics.

    Args:
        coalesced_df: Input product-level metrics DataFrame.
        profit_col: Column name representing gross profit / revenue to use.
        lead_col: Column name for lead indicator (falls back to isLead/main_product_flag).
        weight_col: Column name used for lead-only allocation weights.
        column_overrides: Optional mapping of standardized column -> source column.

    Returns:
        MetricsResult with ad_data, sku_allocation, sku_performance DataFrames.
    """
    if coalesced_df.empty:
        empty = pd.DataFrame()
        return MetricsResult(ad_data=empty, sku_allocation=empty, sku_performance=empty)

    column_map = infer_column_map(coalesced_df, overrides=column_overrides)
    df = apply_column_map(coalesced_df, column_map)
    df = ensure_date_column(df, "date")

    if profit_col in coalesced_df.columns and "grossProfit" not in df.columns:
        df["grossProfit"] = pd.to_numeric(coalesced_df[profit_col], errors="coerce").fillna(0.0)

    required = ["date", "platform", "campaignId", "adSetId", "adId", "productId"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    for name_col in ["campaignName", "adSetName", "adName", "productGroupName", "productName"]:
        if name_col not in df.columns:
            df[name_col] = ""

    if "productGroupId" not in df.columns:
        df["productGroupId"] = ""

    if "match_stage" in coalesced_df.columns and "match_stage" not in df.columns:
        df["match_stage"] = coalesced_df["match_stage"].fillna("unmatched").astype(str)

    for id_col in ["platform", "campaignId", "adSetId", "adId", "productId", "productGroupId"]:
        df[id_col] = df[id_col].astype(str)

    for metric in ["spend", "impressions", "clicks", "interactions", "conversions", "grossProfit"]:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        else:
            df[metric] = 0.0

    df["isLead"] = _normalize_lead_flag(df, lead_col)
    df["sku_weight"] = _select_weight_column(df, weight_col)

    dims = [
        "date",
        "productId",
        "productGroupId",
        "productGroupName",
        "productName",
        "platform",
        "campaignId",
        "campaignName",
        "adSetId",
        "adSetName",
        "adId",
        "adName",
        "isLead",
        "match_stage",
    ]
    dims = [d for d in dims if d in df.columns]

    metric_cols = ["spend", "impressions", "clicks", "grossProfit", "sku_weight"]
    sku_alloc = df.groupby(dims, dropna=False)[metric_cols].sum().reset_index()

    group_keys = ["date", "platform", "campaignId", "adSetId", "adId"]
    totals = sku_alloc.groupby(group_keys, dropna=False).agg(
        ad_spend_total=("spend", _sum_safe),
        ad_impressions_total=("impressions", _sum_safe),
        ad_clicks_total=("clicks", _sum_safe),
        ad_gross_profit_total=("grossProfit", _sum_safe),
        total_weight_all=("sku_weight", _sum_safe),
    ).reset_index()

    sku_alloc = sku_alloc.merge(totals, on=group_keys, how="left")

    sku_alloc["lead_weight"] = np.where(sku_alloc["isLead"] == 1, sku_alloc["sku_weight"], 0.0)
    sku_alloc["total_weight_lead"] = sku_alloc.groupby(group_keys, dropna=False)["lead_weight"].transform("sum")

    sku_alloc["share"] = np.where(
        sku_alloc["total_weight_all"] > 0,
        sku_alloc["sku_weight"] / sku_alloc["total_weight_all"],
        0.0,
    )

    sku_alloc["share_lead_only"] = np.where(
        sku_alloc["total_weight_lead"] > 0,
        sku_alloc["lead_weight"] / sku_alloc["total_weight_lead"],
        0.0,
    )

    for col, total_col in [
        ("spend_lead_only", "ad_spend_total"),
        ("impressions_lead_only", "ad_impressions_total"),
        ("clicks_lead_only", "ad_clicks_total"),
        ("gross_profit_lead_only", "ad_gross_profit_total"),
    ]:
        sku_alloc[col] = sku_alloc[total_col] * sku_alloc["share_lead_only"]

    sku_alloc = sku_alloc.rename(
        columns={
            "grossProfit": "gross_profit_fair",
            "spend": "spend_fair",
            "impressions": "impressions_fair",
            "clicks": "clicks_fair",
        }
    )

    if "productGroupId" not in sku_alloc.columns:
        sku_alloc["productGroupId"] = ""
    if "productGroupName" not in sku_alloc.columns:
        sku_alloc["productGroupName"] = ""
    if "productName" not in sku_alloc.columns:
        sku_alloc["productName"] = ""

    sku_allocation_cols = [
        "date",
        "productId",
        "productGroupId",
        "productGroupName",
        "productName",
        "platform",
        "campaignId",
        "campaignName",
        "adSetId",
        "adSetName",
        "adId",
        "adName",
        "isLead",
        "match_stage",
        "sku_weight",
        "share",
        "gross_profit_fair",
        "spend_fair",
        "impressions_fair",
        "clicks_fair",
        "gross_profit_lead_only",
        "spend_lead_only",
        "impressions_lead_only",
        "clicks_lead_only",
    ]
    sku_allocation_cols = [c for c in sku_allocation_cols if c in sku_alloc.columns]
    sku_allocation = sku_alloc[sku_allocation_cols].copy()

    sku_perf = sku_allocation.groupby(["date", "productId"], dropna=False).agg(
        sku_spend_fair=("spend_fair", _sum_safe),
        sku_impressions_fair=("impressions_fair", _sum_safe),
        sku_clicks_fair=("clicks_fair", _sum_safe),
        sku_gross_profit_fair=("gross_profit_fair", _sum_safe),
        sku_spend_lead_only=("spend_lead_only", _sum_safe),
        sku_impressions_lead_only=("impressions_lead_only", _sum_safe),
        sku_clicks_lead_only=("clicks_lead_only", _sum_safe),
        sku_gross_profit_lead_only=("gross_profit_lead_only", _sum_safe),
    ).reset_index()

    dims_cols = ["productId", "productGroupId", "productGroupName", "productName"]
    dims_cols = [c for c in dims_cols if c in sku_allocation.columns]
    if dims_cols:
        dims = sku_allocation[dims_cols].drop_duplicates("productId")
        sku_perf = sku_perf.merge(dims, on="productId", how="left")

    ad_data_cols = [
        "date",
        "platform",
        "campaignId",
        "campaignName",
        "adSetId",
        "adSetName",
        "adId",
        "adName",
        "match_stage",
        "ad_spend_total",
        "ad_impressions_total",
        "ad_clicks_total",
        "ad_gross_profit_total",
    ]
    ad_data_cols = [c for c in ad_data_cols if c in sku_alloc.columns]
    ad_data = sku_alloc[ad_data_cols].drop_duplicates(group_keys)

    return MetricsResult(ad_data=ad_data, sku_allocation=sku_allocation, sku_performance=sku_perf)
