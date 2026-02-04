"""
Metrics coalescing and reconciliation functions with validation and error handling.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple
from .utils import validate_dataframe


METRICS = [
    "spend",
    "impressions",
    "interactions",
    "clicks",
    "conversions",
    "conversionsValue",
]

FULL_GRAIN = [
    "websiteId",
    "campaignId",
    "adSetId",
    "adId",
    "timestamp",
    "countryCode",
    "platform",
]


def validate_metrics_dataframes(df_prod: pd.DataFrame, df_country: pd.DataFrame):
    """
    Validate that the input DataFrames have the required structure for coalescing.

    Args:
        df_prod: Product metrics DataFrame
        df_country: Country metrics DataFrame

    Raises:
        ValueError: If validation fails
    """
    if df_prod is None or df_country is None:
        raise ValueError("Input DataFrames cannot be None")

    # Check that required columns exist in at least one of the dataframes
    required_in_either = ["websiteId", "campaignId", "adSetId", "adId", "timestamp"]
    for col in required_in_either:
        if col not in df_prod.columns and col not in df_country.columns:
            raise ValueError(f"Required column '{col}' not found in either DataFrame")

    # Check that metric columns exist in both dataframes
    missing_metrics_prod = [m for m in METRICS if m not in df_prod.columns]
    missing_metrics_country = [m for m in METRICS if m not in df_country.columns]

    if missing_metrics_prod:
        logging.warning(f"Missing metric columns in df_prod: {missing_metrics_prod}")
    if missing_metrics_country:
        logging.warning(f"Missing metric columns in df_country: {missing_metrics_country}")

    logging.info(f"Validated metrics DataFrames: prod={df_prod.shape}, country={df_country.shape}")


def coalesce_products_base_country_supplement_robust(
    df_prod: pd.DataFrame,
    df_country: pd.DataFrame,
    residual_product_id="__unmapped__",
    tol=1e-9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Robust coalescing of product and country metrics with validation.

    Args:
        df_prod: Product metrics DataFrame
        df_country: Country metrics DataFrame
        residual_product_id: ID to assign to unmapped records
        tol: Tolerance for floating point comparisons

    Returns:
        Tuple of (coalesced DataFrame, orphaned products DataFrame)
    """
    validate_metrics_dataframes(df_prod, df_country)

    logging.info(f"Starting coalescing: prod={df_prod.shape}, country={df_country.shape}")

    prod = df_prod.copy().reset_index(drop=True)
    country = df_country.copy().reset_index(drop=True)

    for df in (prod, country):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for m in METRICS:
            if m in df.columns:
                df[m] = pd.to_numeric(df.get(m, 0), errors="coerce").fillna(0.0)

    grains = [
        FULL_GRAIN,
        [c for c in FULL_GRAIN if c != "adId"],
        [c for c in FULL_GRAIN if c not in ("adId", "adSetId")],
    ]

    # Track which products matched at which grain
    prod["_matched_grain"] = -1  # -1 = unmatched

    country_ledger = country.copy()
    for m in METRICS:
        if m in country_ledger.columns:
            country_ledger[m + "_residual"] = country_ledger[m]

    for i, G in enumerate(grains):
        # Check if all columns in grain exist in both dataframes
        if not all(c in prod.columns for c in G) or not set(G).issubset(set(country_ledger.columns)):
            logging.debug(f"Skipping grain {i} due to missing columns: {G}")
            continue

        # Only process products not yet matched
        unmatched_mask = prod["_matched_grain"] == -1
        prod_unmatched = prod[unmatched_mask]
        if prod_unmatched.empty:
            logging.debug(f"All products matched at grain {i}, stopping early")
            break

        # Aggregate unmatched products at this grain
        try:
            prod_agg = prod_unmatched.groupby(G, as_index=False)[METRICS].sum()
        except Exception as e:
            logging.warning(f"Could not aggregate at grain {i} due to missing metric columns: {e}")
            continue

        # Get country keys at this grain (need to aggregate first to avoid 1:many)
        try:
            country_agg = country_ledger.groupby(G, as_index=False)[[m + "_residual" for m in METRICS if m + "_residual" in country_ledger.columns]].sum()
        except Exception as e:
            logging.warning(f"Could not aggregate country data at grain {i}: {e}")
            continue

        # Find which product keys exist in country
        merged = prod_agg.merge(
            country_agg[G],
            on=G,
            how="inner",
        )

        if merged.empty:
            logging.debug(f"No matches found at grain {i}")
            continue

        # Mark products as matched at this grain
        country_key_set = set(merged[G].apply(tuple, axis=1))
        prod.loc[:, "_tmp_key"] = prod[G].apply(tuple, axis=1)
        newly_matched = unmatched_mask & prod["_tmp_key"].isin(country_key_set)
        prod.loc[newly_matched, "_matched_grain"] = i
        prod.drop(columns=["_tmp_key"], inplace=True)

        # Subtract from country ledger (join prod_agg to country_ledger)
        country_ledger = country_ledger.merge(
            prod_agg,
            on=G,
            how="left",
            suffixes=("", "_prod"),
        )

        for m in METRICS:
            prod_col = f"{m}_prod"
            if prod_col in country_ledger.columns:
                country_ledger[m + "_residual"] -= country_ledger[prod_col].fillna(0)

        country_ledger = country_ledger.drop(columns=[c for c in country_ledger.columns if c.endswith("_prod")])

    # Build residual rows
    supplement = country_ledger[
        [c for c in FULL_GRAIN if c in country_ledger.columns]
        + [m + "_residual" for m in METRICS if m + "_residual" in country_ledger.columns]
    ].copy()

    supplement.rename(columns={m + "_residual": m for m in METRICS if m + "_residual" in supplement.columns}, inplace=True)

    for m in METRICS:
        if m in supplement.columns:
            supplement.loc[supplement[m].between(-tol, 0), m] = 0.0

    nonzero = np.zeros(len(supplement), dtype=bool)
    for m in METRICS:
        if m in supplement.columns:
            nonzero |= supplement[m].abs() > tol
    supplement = supplement[nonzero]

    supplement["productId"] = residual_product_id
    supplement["source"] = "country_supplement"

    # Only output products that matched a country row
    prod_matched = prod[prod["_matched_grain"] >= 0].copy()
    prod_matched["source"] = "products"
    prod_matched.drop(columns=["_matched_grain"], inplace=True)

    # Orphan products (optional: for debugging)
    orphan_spend = prod.loc[prod["_matched_grain"] == -1, "spend"].sum() if "spend" in prod.columns else 0
    if orphan_spend > tol:
        logging.warning(f"{(prod['_matched_grain'] == -1).sum()} orphan products with spend={orphan_spend:,.2f} excluded (no matching country row)")

    final_cols = (
        [c for c in FULL_GRAIN if c in country.columns or c in prod.columns]
        + ["productId"]
        + METRICS
        + ["source"]
    )

    # Only include columns that actually exist in the dataframes
    final_cols = [c for c in final_cols if c in prod_matched.columns or c in supplement.columns]

    coalesced = pd.concat([prod_matched[final_cols], supplement[final_cols]], ignore_index=True)

    logging.info(f"Coalescing completed: result={coalesced.shape}")

    orphaned_products = prod[prod["_matched_grain"] == -1] if len(prod[prod["_matched_grain"] == -1]) > 0 else pd.DataFrame()

    return coalesced, orphaned_products