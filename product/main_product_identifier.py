"""
Functions for identifying main products targeted by campaigns.
"""

import pandas as pd
import logging


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
    
    # Create a mapping from (campaignId, adSetId, adId) to productGroupIds_targeted list
    targeting_map = {}
    for _, row in final_targeting.iterrows():
        key = (row['campaignId'], row['adSetId'], row['adId'])
        targeting_map[key] = row['productGroupIds_targeted']
    
    # Iterate through coalesced_df and set the flag where appropriate
    for idx, row in result_df.iterrows():
        key = (row['campaignId'], row['adSetId'], row['adId'])
        
        if key in targeting_map:
            targeted_groups = targeting_map[key]
            product_group_id = row['productGroupId_']
            
            # Check if productGroupId_ is in the targeted list
            if product_group_id in targeted_groups:
                result_df.at[idx, 'main_product_flag'] = 1
    
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

    # Explode the list column to create one row per product group ID
    try:
        exploded_targeting = final_targeting_filtered.explode('productGroupIds_targeted')
        # Remove rows where productGroupIds_targeted became NaN after explode
        exploded_targeting = exploded_targeting[exploded_targeting['productGroupIds_targeted'].notna()].copy()
        exploded_targeting = exploded_targeting.rename(columns={'productGroupIds_targeted': 'productGroupId_'})

        # Create a temporary identifier for matching
        targeting_keys = exploded_targeting[['campaignId', 'adSetId', 'adId', 'productGroupId_']].drop_duplicates()

        if not targeting_keys.empty:
            # Create a merge key in coalesced_df
            coalesced_with_flag = result_df.merge(
                targeting_keys,
                on=['campaignId', 'adSetId', 'adId', 'productGroupId_'],
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