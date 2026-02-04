"""
URL beautification and processing functions with validation and memory management.
"""

import pandas as pd
import numpy as np
import ast
import logging
from urllib.parse import urlparse, parse_qs, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from .utils import validate_dataframe, memory_monitor


# ---------------------------
# Constants
# ---------------------------
UNWRAP_PARAM_CANDIDATES = frozenset([
    "url", "u", "q", "target", "dest", "destination", "redir", "redirect", "redirect_url",
    "redirectUri", "redirect_uri", "r", "to", "out", "continue", "next", "return", "returnTo",
    "return_to", "callback", "cb", "deep_link", "deeplink", "dl", "link"
])

NA_VALUES = frozenset(["", "<NA>", "NA", "NaN", "nan", "None"])
URL_PREFIXES = ("http://", "https://", "www.")


# ---------------------------
# Optimized URL beautifier with caching
# ---------------------------
@lru_cache(maxsize=50000)
def beautify_url(url: str, max_depth: int = 5) -> str:
    """Extract real destination URL from redirect wrappers. Cached."""
    if not url or not isinstance(url, str):
        return url

    current = url.strip()
    if not current:
        return current

    for _ in range(max_depth):
        try:
            parsed = urlparse(current)
            qs = parse_qs(parsed.query, keep_blank_values=False)

            found = _find_url_in_params(qs)

            if not found and parsed.fragment:
                frag_qs = parse_qs(parsed.fragment, keep_blank_values=False)
                found = _find_url_in_params(frag_qs)

            if found and found != current:
                current = found
                continue

            return current
        except Exception:
            return current

    return current


def _find_url_in_params(params: dict) -> str | None:
    """Find URL in query params dict."""
    for key in UNWRAP_PARAM_CANDIDATES:
        if key in params and params[key]:
            candidate = unquote(params[key][0]).strip()
            if candidate.startswith(URL_PREFIXES):
                return candidate
    return None


# ---------------------------
# Cleaning functions
# ---------------------------
def clean_destination_url(val):
    """Clean single destination URL value."""
    if pd.isna(val):
        return None

    if isinstance(val, list):
        return val

    if not isinstance(val, str):
        return val

    s = val.strip()

    if s in NA_VALUES:
        return None

    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return s

    return s


def to_list(val):
    """Convert value to list."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return val
    return [val]


# ---------------------------
# Parallel URL beautification
# ---------------------------
def beautify_urls_parallel(urls: pd.Series, max_workers: int = 8) -> pd.Series:
    """Parallel URL beautification for large datasets."""
    unique_urls = urls.dropna().unique()
    unique_urls = [u for u in unique_urls if isinstance(u, str) and u.strip()]

    if len(unique_urls) < 1000:
        return urls.apply(lambda x: beautify_url(x) if pd.notna(x) and isinstance(x, str) else x)

    url_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(beautify_url, url): url for url in unique_urls}
        for future in as_completed(futures):
            original = futures[future]
            try:
                url_map[original] = future.result()
            except Exception as e:
                logging.warning(f"Error processing URL {original}: {e}")
                url_map[original] = original

    return urls.apply(lambda x: url_map.get(x, x))


# ---------------------------
# Main pipeline
# ---------------------------
def process_destination_urls(df_ads_ori: pd.DataFrame, parallel: bool = True, max_workers: int = 8, min_url_length: int = 5) -> tuple:
    """
    Optimized URL processing pipeline with validation and memory management.

    Args:
        df_ads_ori: Original ads DataFrame
        parallel: Whether to use parallel processing
        max_workers: Number of workers for parallel processing
        min_url_length: Minimum length for a URL to be considered valid

    Returns:
        df_ads_exploded: Exploded rows with beautified URLs
        df_ads_unique_urls: Unique URL summary with weighted metrics

    Columns in df_ads_exploded:
        - platform (if exists)
        - destinationUrl_original: completely unaltered original value
        - destinationUrl_item: individual URL after exploding lists
        - destinationUrl_beautified: unwrapped/cleaned URL
        - destinationUrl_count: number of URLs in original row
        - destinationUrl_multiplier: same as count (for weighting)
        - row_id: original row index
    """
    # Validate input DataFrame
    validate_dataframe(df_ads_ori, ["destinationUrl"], "df_ads_ori")

    logging.info(f"Original rows: {len(df_ads_ori):,}")

    # Work on minimal columns only
    cols = ["platform", "destinationUrl"] if "platform" in df_ads_ori.columns else ["destinationUrl"]
    df = df_ads_ori[cols].copy()
    df["row_id"] = np.arange(len(df), dtype=np.int32)

    # Keep original unaltered
    df["destinationUrl_original"] = df["destinationUrl"]

    # Clean URLs
    df["destinationUrl_clean"] = df["destinationUrl"].apply(clean_destination_url)

    # Convert to lists
    df["destinationUrl_list"] = df["destinationUrl_clean"].apply(to_list)
    df["destinationUrl_count"] = df["destinationUrl_list"].apply(len).astype(np.int16)

    # Explode
    df_exploded = df.explode("destinationUrl_list", ignore_index=True)
    df_exploded = df_exploded.rename(columns={"destinationUrl_list": "destinationUrl_item"})

    # Filter empty/null
    mask = df_exploded["destinationUrl_item"].notna()
    df_exploded = df_exploded.loc[mask].copy()

    str_mask = df_exploded["destinationUrl_item"].astype(str).str.len() >= min_url_length
    df_exploded = df_exploded.loc[str_mask].copy()

    logging.info(f"Exploded rows: {len(df_exploded):,}")

    # Memory monitoring
    memory_monitor()

    # Beautify URLs
    if parallel and len(df_exploded) > 1000:
        df_exploded["destinationUrl_beautified"] = beautify_urls_parallel(
            df_exploded["destinationUrl_item"], max_workers=max_workers
        )
    else:
        df_exploded["destinationUrl_beautified"] = df_exploded["destinationUrl_item"].apply(
            lambda x: beautify_url(x) if isinstance(x, str) else x
        )

    df_exploded["destinationUrl_multiplier"] = df_exploded["destinationUrl_count"]

    # Unique URL aggregation
    df_unique = (
        df_exploded
        .groupby("destinationUrl_beautified", as_index=False)
        .agg(
            rows_using_url=("row_id", "nunique"),
            occurrences=("row_id", "size"),
            weighted_multiplier_sum=("destinationUrl_multiplier", "sum"),
            max_multiplier=("destinationUrl_multiplier", "max"),
        )
    )

    # Optimize memory
    for col in ["rows_using_url", "occurrences", "weighted_multiplier_sum"]:
        df_unique[col] = df_unique[col].astype(np.int32)
    df_unique["max_multiplier"] = df_unique["max_multiplier"].astype(np.int16)

    logging.info(f"Unique beautified URLs: {len(df_unique):,}")

    # Drop intermediate columns, keep original
    df_exploded = df_exploded.drop(columns=["destinationUrl", "destinationUrl_clean"], errors="ignore")

    # Reorder columns for clarity
    col_order = ["row_id", "platform", "destinationUrl_original", "destinationUrl_item",
                 "destinationUrl_beautified", "destinationUrl_count", "destinationUrl_multiplier"]
    col_order = [c for c in col_order if c in df_exploded.columns]
    df_exploded = df_exploded[col_order]

    logging.info(f"URL cache stats: {beautify_url.cache_info()}")

    # Memory monitoring
    memory_monitor()

    return df_exploded, df_unique