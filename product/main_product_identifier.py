"""
Functions for identifying main products targeted by campaigns.
"""

import pandas as pd
import logging
from typing import Optional


def _resolve_product_group_column(coalesced_df: pd.DataFrame) -> Optional[str]:
    """Return the product group id column name from coalesced_df, if present."""
    candidates = ("productGroupId_", "productGroupId", "productGroupId_x", "productGroupId_y")
    for candidate in candidates:
        if candidate in coalesced_df.columns:
            return candidate
    return None


def identify_main_products(final_targeting: pd.DataFrame, coalesced_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify main products targeted by campaigns by merging final_targeting with coalesced_df.
    
    This function creates a 'main_product_flag' column in coalesced_df that indicates whether
    the productGroupId_ matches one of the elements in the productGroupIds_targeted list
    for the corresponding campaignId, adSetId, adId combination.
    
    Args:
        final_targeting: DataFrame with columns ['campaignId', 'adSetId', 'adId', 'productGroupIds_targeted']
        coalesced_df: DataFrame with columns ['campaignId', 'adSetId', 'adId', 'productGroupId_']
        
    Returns:
        Updated coalesced_df with 'main_product_flag' column (1 if matched, 0 otherwise)
    """
    logging.info(f"Identifying main products. Final targeting shape: {final_targeting.shape}, Coalesced DF shape: {coalesced_df.shape}")
    
    # Create a copy of coalesced_df to avoid modifying the original
    result_df = coalesced_df.copy()
    
    # Initialize the main_product_flag column to 0
    result_df['main_product_flag'] = 0

    product_group_column = _resolve_product_group_column(result_df)
    if product_group_column is None:
        logging.warning(
            "No product group id column found in coalesced_df. "
            "Expected one of: productGroupId_, productGroupId, productGroupId_x, productGroupId_y."
        )
        return result_df
    
    # Create a mapping from (campaignId, adSetId, adId) to productGroupIds_targeted list
    targeting_map = {}
    match_stage_map = {}
    for _, row in final_targeting.iterrows():
        key = (row['campaignId'], row['adSetId'], row['adId'])
        targeting_map[key] = row['productGroupIds_targeted']
        if "match_stage" in row:
            match_stage_map[key] = row["match_stage"]
    
    # Iterate through coalesced_df and set the flag where appropriate
    for idx, row in result_df.iterrows():
        key = (row['campaignId'], row['adSetId'], row['adId'])
        
        if key in targeting_map:
            targeted_groups = targeting_map[key]
            product_group_id = row[product_group_column]

            # Check if productGroupId_ is in the targeted list
            if product_group_id in targeted_groups:
                result_df.at[idx, 'main_product_flag'] = 1
        if match_stage_map:
            result_df.at[idx, "match_stage"] = match_stage_map.get(key, "unmatched")
    
    logging.info(f"Added main_product_flag to {result_df[result_df['main_product_flag'] == 1].shape[0]} rows")
    
    return result_df


def identify_main_products_vectorized(final_targeting: pd.DataFrame, coalesced_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized version of identify_main_products for better performance on large datasets.

    Args:
        final_targeting: DataFrame with columns ['campaignId', 'adSetId', 'adId', 'productGroupIds_targeted']
        coalesced_df: DataFrame with columns ['campaignId', 'adSetId', 'adId', 'productGroupId_']

    Returns:
        Updated coalesced_df with 'main_product_flag' column (1 if matched, 0 otherwise)
    """
    logging.info(f"Identifying main products (vectorized). Final targeting shape: {final_targeting.shape}, Coalesced DF shape: {coalesced_df.shape}")

    # Create a copy of coalesced_df to avoid modifying the original
    result_df = coalesced_df.copy()

    # Initialize the main_product_flag column to 0
    result_df['main_product_flag'] = 0

    product_group_column = _resolve_product_group_column(result_df)
    if product_group_column is None:
        logging.warning(
            "No product group id column found in coalesced_df. "
            "Expected one of: productGroupId_, productGroupId, productGroupId_x, productGroupId_y."
        )
        return result_df

    # Handle empty dataframes
    if final_targeting.empty or coalesced_df.empty:
        logging.info("One or both dataframes are empty, returning original coalesced_df with flags set to 0")
        return result_df

    # Explode the productGroupIds_targeted column to create individual rows
    # First, we need to make sure the column contains lists
    final_targeting_expanded = final_targeting.copy()
    final_targeting_expanded['productGroupIds_targeted'] = final_targeting_expanded['productGroupIds_targeted'].apply(
        lambda x: x if isinstance(x, list) and x else []  # Return empty list if x is None or not a list
    )

    # Only proceed if there are any non-empty lists
    non_empty_mask = final_targeting_expanded['productGroupIds_targeted'].apply(len) > 0
    if not non_empty_mask.any():
        logging.info("No non-empty productGroupIds_targeted lists found, returning original coalesced_df with flags set to 0")
        return result_df

    # Filter to only rows with non-empty lists
    final_targeting_filtered = final_targeting_expanded[non_empty_mask].copy()

    if final_targeting_filtered.empty:
        logging.info("No rows with non-empty productGroupIds_targeted lists after filtering, returning original coalesced_df with flags set to 0")
        return result_df

    if "match_stage" in final_targeting_filtered.columns:
        match_stage_lookup = final_targeting_filtered[["campaignId", "adSetId", "adId", "match_stage"]].drop_duplicates()
        result_df = result_df.merge(match_stage_lookup, on=["campaignId", "adSetId", "adId"], how="left")
        result_df["match_stage"] = result_df["match_stage"].fillna("unmatched").astype(str)

    # Explode the list column to create one row per product group ID
    try:
        exploded_targeting = final_targeting_filtered.explode('productGroupIds_targeted')
        # Remove rows where productGroupIds_targeted became NaN after explode
        exploded_targeting = exploded_targeting[exploded_targeting['productGroupIds_targeted'].notna()].copy()
        exploded_targeting = exploded_targeting.rename(columns={'productGroupIds_targeted': product_group_column})

        # Create a temporary identifier for matching
        targeting_keys = exploded_targeting[['campaignId', 'adSetId', 'adId', product_group_column]].drop_duplicates()

        if not targeting_keys.empty:
            # Create a merge key in coalesced_df
            coalesced_with_flag = result_df.merge(
                targeting_keys,
                on=['campaignId', 'adSetId', 'adId', product_group_column],
                how='left',
                indicator=True
            )

            # Set the flag to 1 where there was a match
            result_df['main_product_flag'] = (coalesced_with_flag['_merge'] == 'both').astype(int)
    except Exception as e:
        logging.warning(f"Error in vectorized matching: {e}. Falling back to non-vectorized method.")
        # Fallback to the non-vectorized method
        return identify_main_products(final_targeting, coalesced_df)

    logging.info(f"Added main_product_flag to {result_df[result_df['main_product_flag'] == 1].shape[0]} rows")

    return result_df
