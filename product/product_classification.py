"""
Product ID classification and matching functions with validation and error handling.
"""

import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
try:
    import ahocorasick
    HAS_AHOCORASICK = True
except ImportError:
    HAS_AHOCORASICK = False

from .utils import validate_dataframe


def _process_chunk(args):
    source_chunk, A, prefer_prod, start_idx = args

    prod_out = [None] * len(source_chunk)
    group_out = [None] * len(source_chunk)

    for i, text in enumerate(source_chunk):
        # Ensure string
        if text is None:
            text = ""

        best_prod = None
        best_group = None
        best_prod_len = 0
        best_group_len = 0

        # A.iter(text) yields: (end_index, value)
        # where value is (tok, is_prod)
        try:
            for _, (tok, is_prod) in A.iter(text):
                L = len(tok)
                if is_prod:
                    if L > best_prod_len:
                        best_prod = tok
                        best_prod_len = L
                else:
                    if L > best_group_len:
                        best_group = tok
                        best_group_len = L
        except Exception as e:
            logging.warning(f"Error processing text in Aho-Corasick automaton: {e}")
            continue

        # If both matched, apply preference
        if best_prod is not None and best_group is not None:
            if prefer_prod:
                best_group = None
            else:
                best_prod = None

        prod_out[i] = best_prod
        group_out[i] = best_group

    return start_idx, prod_out, group_out


def validate_product_classification_inputs(
    coalesced_df: pd.DataFrame,
    df_prod: pd.DataFrame,
    source_col: str,
    prefer: str
):
    """
    Validate inputs for product classification function.

    Args:
        coalesced_df: Coalesced DataFrame
        df_prod: Product DataFrame
        source_col: Source column name
        prefer: Preference for matching

    Raises:
        ValueError: If validation fails
    """
    if not HAS_AHOCORASICK:
        raise ImportError("ahocorasick is required for this function. Install with 'pip install pyahocorasick'")

    if prefer not in ("productId", "productGroupId"):
        raise ValueError("prefer must be 'productId' or 'productGroupId'")

    if source_col not in coalesced_df.columns:
        raise KeyError(f"source_col '{source_col}' not found in coalesced_df")

    # Validate required columns exist
    validate_dataframe(coalesced_df, [source_col], "coalesced_df")
    validate_dataframe(df_prod, ["productId", "productGroupId"], "df_prod")

    logging.info(f"Validated classification inputs: coalesced_df={coalesced_df.shape}, df_prod={df_prod.shape}")


def classify_product_id_tokens_parallel(
    coalesced_df: pd.DataFrame,
    df_prod: pd.DataFrame,
    source_col: str = "productId_",
    out_product_id_col: str = "productId",
    out_product_group_col: str = "productGroupId",
    prefer: str = "productId",   # "productId" or "productGroupId"
    n_workers: int = 4,
    chunk_size: int = 10000,
    fill_group_from_product: bool = True,  # <- your requested behavior
) -> pd.DataFrame:
    """
    Classify product ID tokens with validation and error handling.

    Args:
        coalesced_df: Coalesced DataFrame with product IDs to classify
        df_prod: Product DataFrame with reference IDs
        source_col: Column name containing source product IDs
        out_product_id_col: Output column name for product IDs
        out_product_group_col: Output column name for product group IDs
        prefer: Whether to prefer product ID or product group ID
        n_workers: Number of workers for parallel processing
        chunk_size: Size of chunks for processing
        fill_group_from_product: Whether to fill group IDs from product IDs

    Returns:
        DataFrame with classified product IDs
    """
    validate_product_classification_inputs(coalesced_df, df_prod, source_col, prefer)

    logging.info(f"Starting product classification: prefer={prefer}, source_col={source_col}")

    # --- Build token sets ---
    prod_ids = (
        df_prod.get("productId", pd.Series(dtype=object))
        .dropna()
        .astype(str)
    )
    group_ids = (
        df_prod.get("productGroupId", pd.Series(dtype=object))
        .dropna()
        .astype(str)
    )

    prod_ids = set(prod_ids.tolist()) - {""}
    group_ids = set(group_ids.tolist()) - {""}

    logging.info(f"Built token sets: {len(prod_ids)} product IDs, {len(group_ids)} group IDs")

    # --- Build Aho–Corasick automaton ---
    A = ahocorasick.Automaton()
    for tok in prod_ids:
        A.add_word(tok, (tok, True))
    for tok in group_ids:
        A.add_word(tok, (tok, False))
    A.make_automaton()

    # --- Prepare source array ---
    source = coalesced_df[source_col].fillna("").astype(str).to_numpy()
    n = len(source)
    prefer_prod = (prefer == "productId")

    # --- Chunk work ---
    chunks = [
        (source[i:i + chunk_size], A, prefer_prod, i)
        for i in range(0, n, chunk_size)
    ]

    prod_out = np.empty(n, dtype=object)
    group_out = np.empty(n, dtype=object)

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for start_idx, prod_res, group_res in ex.map(_process_chunk, chunks):
                end_idx = start_idx + len(prod_res)
                prod_out[start_idx:end_idx] = prod_res
                group_out[start_idx:end_idx] = group_res
    except Exception as e:
        logging.error(f"Error during parallel processing: {e}")
        raise

    out = coalesced_df.copy()
    out[out_product_id_col] = pd.Series(prod_out, index=out.index, dtype="object")
    out[out_product_group_col] = pd.Series(group_out, index=out.index, dtype="object")

    # Normalize None -> NA for easier masking
    out[out_product_id_col] = out[out_product_id_col].where(out[out_product_id_col].notna(), pd.NA)
    out[out_product_group_col] = out[out_product_group_col].where(out[out_product_group_col].notna(), pd.NA)

    # --- Your requested post-fill:
    # If productGroupId is null and productId is not null,
    # look up df_prod to fill productGroupId.
    if fill_group_from_product:
        logging.info("Filling product group IDs from product IDs")
        # Build mapping productId -> productGroupId
        # If duplicates exist, we take the first non-null group id encountered.
        map_df = df_prod[["productId", "productGroupId"]].dropna(subset=["productId"])
        map_df["productId"] = map_df["productId"].astype(str)
        map_df["productGroupId"] = map_df["productGroupId"].astype(str)

        # If one productId maps to multiple groupIds, pick the first (can tweak if needed)
        pid_to_gid = (
            map_df.dropna(subset=["productGroupId"])
                  .drop_duplicates(subset=["productId"], keep="first")
                  .set_index("productId")["productGroupId"]
        )

        mask = out[out_product_group_col].isna() & out[out_product_id_col].notna()
        out.loc[mask, out_product_group_col] = out.loc[mask, out_product_id_col].map(pid_to_gid)

    logging.info(f"Product classification completed: result shape={out.shape}")
    return out