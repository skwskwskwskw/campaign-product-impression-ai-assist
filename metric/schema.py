"""Schema helpers for coalesced product metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd


COLUMN_CANDIDATES = {
    "date": ["date", "timestamp", "day"],
    "platform": ["platform"],
    "campaignId": ["campaignId", "campaign_id"],
    "campaignName": ["campaignName", "campaign_name"],
    "adSetId": ["adSetId", "adsetId", "ad_set_id", "adSet_id"],
    "adSetName": ["adSetName", "ad_set_name"],
    "adId": ["adId", "ad_id"],
    "adName": ["adName", "ad_name"],
    "productId": ["productId", "productId_", "product_id"],
    "productGroupId": ["productGroupId", "productGroupId_", "product_group_id"],
    "productName": ["productName", "product_name"],
    "productGroupName": ["productGroupName", "product_group_name"],
    "isLead": ["isLead", "main_product_flag", "lead_flag"],
    "spend": ["spend"],
    "impressions": ["impressions"],
    "interactions": ["interactions"],
    "clicks": ["clicks"],
    "conversions": ["conversions"],
    "grossProfit": ["grossProfit", "gross_profit", "conversionsValue", "revenue"],
    "sku_weight": ["sku_weight", "quantity", "qty", "units"],
}


def _first_match(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_set = set(columns)
    for candidate in candidates:
        if candidate in col_set:
            return candidate
    return None


def infer_column_map(df: pd.DataFrame, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Infer a mapping of source column -> standardized column name.

    Args:
        df: Input DataFrame.
        overrides: Optional mapping of standardized column name -> source column.

    Returns:
        Mapping of source column names to standardized column names.
    """
    overrides = overrides or {}
    mapping: Dict[str, str] = {}

    for standard_name, candidates in COLUMN_CANDIDATES.items():
        if standard_name in overrides and overrides[standard_name]:
            source_name = overrides[standard_name]
        else:
            source_name = _first_match(df.columns, candidates)

        if source_name:
            mapping[source_name] = standard_name

    return mapping


def apply_column_map(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    """Return a copy of df with columns renamed using the provided map."""
    return df.rename(columns=column_map).copy()


def ensure_date_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Ensure the date column exists and is converted to pandas datetime (date)."""
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column '{date_col}'.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    return df
