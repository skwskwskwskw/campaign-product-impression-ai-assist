"""
Ad-to-Product Matcher: Core logic for identifying which products each ad targets.

This module provides the main functionality to:
1. Match ads to products using URL patterns, product names, and fuzzy matching
2. Identify "lead" products (directly targeted) vs "halo" products (spillover)
3. Allocate ad metrics (spend, impressions) to products fairly

Usage:
    from simple.ad_product_matcher import AdProductMatcher

    matcher = AdProductMatcher(config)
    results = matcher.run(df_ads, df_products, df_metrics)
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from urllib.parse import urlparse, unquote
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Try to use rapidfuzz for fast fuzzy matching, fall back to difflib
try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False

from .config import PipelineConfig


# ============================================================
# Data Classes for Results
# ============================================================

@dataclass
class MatchResult:
    """Result of matching an ad to products."""
    product_group_ids: List[str]
    match_stage: str  # exact_url, exact_product, fuzzy, token_overlap, unmatched
    confidence_score: float = 1.0


@dataclass
class TargetingResult:
    """Full targeting result for all ads."""
    targeting_df: pd.DataFrame      # Ad -> ProductGroup mappings
    metrics_df: pd.DataFrame        # Metrics with lead/halo flags
    debug_df: Optional[pd.DataFrame] = None  # Intermediate matching details


# ============================================================
# Helper Functions
# ============================================================

def safe_str(x) -> str:
    """Safely convert any value to string."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (ValueError, TypeError):
        pass
    return str(x).strip()


def normalize_text(s: str) -> str:
    """Normalize text for matching: lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]+", "", safe_str(s).lower())


def tokenize(s: str, min_length: int = 3) -> List[str]:
    """Split text into tokens for matching."""
    s = safe_str(s).lower()
    if not s:
        return []
    tokens = re.findall(r"[a-z0-9]+", s)
    return [t for t in tokens if len(t) >= min_length]


def canonical_url(url: str) -> str:
    """Normalize URL for comparison: host + path, decoded, no trailing slash."""
    url = safe_str(url)
    if not url:
        return ""
    url = unquote(url)
    if "://" not in url and url.startswith("www."):
        url = "https://" + url
    parsed = urlparse(url)
    host = safe_str(parsed.netloc).lower()
    path = re.sub(r"/{2,}", "/", safe_str(parsed.path)).rstrip("/")
    return f"{host}{path}" if (host or path) else ""


def extract_url_parts(url: str) -> Tuple[str, str]:
    """
    Extract collection and product slugs from URL.

    Returns:
        Tuple of (collection_slug, product_slug)
    """
    url = safe_str(url)
    collection = ""
    product = ""

    collection_match = re.search(r"/collections?/([A-Za-z0-9_-]+)", url)
    if collection_match:
        collection = collection_match.group(1).lower()

    product_match = re.search(r"/products/([A-Za-z0-9_-]+)", url)
    if product_match:
        product = product_match.group(1).lower()

    return collection, product


# ============================================================
# Main Matcher Class
# ============================================================

class AdProductMatcher:
    """
    Matches ads to products to identify which products each ad is targeting.

    The matching process follows these stages (in order of confidence):
    1. Exact URL match: Ad destination URL matches product URL
    2. Exact product slug: /products/X in ad URL matches product
    3. Fuzzy matching: Ad name/description fuzzy matches product name
    4. Token overlap: Shared keywords between ad and product

    Products matched at stages 1-3 are considered "lead" products.
    Products matched only at stage 4 are typically "halo" (spillover).
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(__name__)

        # Indices built from product data
        self._url_index: Dict[Tuple[str, str], List[str]] = {}
        self._product_slug_index: Dict[Tuple[str, str], List[str]] = {}
        self._fuzzy_corpus_by_name: Dict[str, Tuple[List[str], List[str]]] = {}
        self._token_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        # Cache for fuzzy results
        self._fuzzy_cache: Dict[tuple, List[Tuple[str, int]]] = {}

    def build_product_index(self, df_products: pd.DataFrame) -> None:
        """
        Build search indices from product catalog.

        Args:
            df_products: DataFrame with columns:
                - websiteId
                - productGroupId
                - name (product name)
                - url (product URL)
        """
        self.logger.info(f"Building product index from {len(df_products)} products")

        # Clear existing indices
        self._url_index.clear()
        self._product_slug_index.clear()
        self._fuzzy_corpus_by_name.clear()
        self._token_index.clear()
        self._fuzzy_cache.clear()

        # Normalize columns
        df = df_products.copy()
        df["websiteId"] = df["websiteId"].apply(safe_str)
        df["productGroupId"] = df["productGroupId"].apply(safe_str)
        df["name_norm"] = df["name"].apply(normalize_text)
        df["url_canon"] = df.get("url", pd.Series([""] * len(df))).apply(canonical_url)

        # Extract product slugs from URLs
        df["product_slug"] = df.get("url", pd.Series([""] * len(df))).apply(
            lambda u: extract_url_parts(u)[1]
        )

        # Build URL index: (websiteId, canonical_url) -> [productGroupIds]
        for (wid, url), grp in df[df["url_canon"] != ""].groupby(["websiteId", "url_canon"]):
            self._url_index[(wid, url)] = list(grp["productGroupId"].unique())

        # Build product slug index: (websiteId, slug) -> [productGroupIds]
        for (wid, slug), grp in df[df["product_slug"] != ""].groupby(["websiteId", "product_slug"]):
            self._product_slug_index[(wid, slug)] = list(grp["productGroupId"].unique())

        # Build fuzzy corpus by website
        if HAS_RAPIDFUZZ:
            for wid, grp in df.groupby("websiteId"):
                names = grp["name_norm"].tolist()
                pgids = grp["productGroupId"].tolist()
                self._fuzzy_corpus_by_name[wid] = (names, pgids)

        # Build token index for fallback matching
        for _, row in df.iterrows():
            wid = row["websiteId"]
            pgid = row["productGroupId"]
            tokens = set(tokenize(row["name_norm"], self.config.min_token_length))
            for token in tokens:
                self._token_index[(wid, token)].add(pgid)

        self.logger.info(
            f"Index built: {len(self._url_index)} URLs, "
            f"{len(self._product_slug_index)} slugs, "
            f"{len(self._fuzzy_corpus_by_name)} fuzzy corpora"
        )

    def _fuzzy_match(self, website_id: str, query: str) -> List[Tuple[str, int]]:
        """
        Find products matching query using fuzzy string matching.

        Returns:
            List of (productGroupId, score) tuples, sorted by score descending
        """
        query = normalize_text(query)
        if not query:
            return []

        cache_key = ("fuzzy", website_id, query)
        if cache_key in self._fuzzy_cache:
            return self._fuzzy_cache[cache_key]

        results = []

        if HAS_RAPIDFUZZ and website_id in self._fuzzy_corpus_by_name:
            names, pgids = self._fuzzy_corpus_by_name[website_id]
            if names:
                try:
                    matches = rf_process.extract(
                        query,
                        names,
                        scorer=fuzz.ratio,
                        score_cutoff=self.config.fuzzy_threshold,
                        limit=self.config.fuzzy_limit
                    )
                    # De-duplicate by productGroupId, keeping best score
                    best = {}
                    for _, score, idx in matches:
                        pgid = pgids[idx]
                        if pgid not in best or score > best[pgid]:
                            best[pgid] = int(score)
                    results = sorted(best.items(), key=lambda x: (-x[1], x[0]))
                except Exception as e:
                    self.logger.warning(f"Fuzzy matching error: {e}")

        self._fuzzy_cache[cache_key] = results
        return results

    def _token_overlap_match(self, website_id: str, query: str) -> List[str]:
        """
        Find products with overlapping tokens (fallback matching).

        Returns:
            List of productGroupIds sorted by overlap count descending
        """
        query_tokens = set(tokenize(query, self.config.min_token_length))
        if not query_tokens:
            return []

        scores = defaultdict(int)
        for token in query_tokens:
            for pgid in self._token_index.get((website_id, token), set()):
                scores[pgid] += 1

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [pgid for pgid, _ in ranked][:self.config.max_products_per_ad]

    def match_ad(
        self,
        website_id: str,
        destination_url: str,
        ad_name: str = "",
        collections: str = "",
    ) -> MatchResult:
        """
        Match a single ad to products.

        Args:
            website_id: Website identifier
            destination_url: Ad's destination URL
            ad_name: Ad name/title
            collections: Collection identifier from URL

        Returns:
            MatchResult with matched product IDs and match stage
        """
        max_products = self.config.max_products_per_ad

        # Stage 1: Exact URL match
        url_canon = canonical_url(destination_url)
        if url_canon:
            pgids = self._url_index.get((website_id, url_canon), [])
            if pgids:
                return MatchResult(
                    product_group_ids=pgids[:max_products],
                    match_stage="exact_url",
                    confidence_score=1.0
                )

        # Stage 2: Product slug match
        _, product_slug = extract_url_parts(destination_url)
        if product_slug:
            pgids = self._product_slug_index.get((website_id, product_slug), [])
            if pgids:
                return MatchResult(
                    product_group_ids=pgids[:max_products],
                    match_stage="exact_product",
                    confidence_score=0.95
                )

        # Stage 3: Fuzzy matching on combined query
        query_parts = [ad_name, collections, product_slug]
        query = " ".join([safe_str(p) for p in query_parts if safe_str(p)])

        if query:
            fuzzy_matches = self._fuzzy_match(website_id, query)
            if fuzzy_matches:
                avg_score = sum(s for _, s in fuzzy_matches) / len(fuzzy_matches)
                return MatchResult(
                    product_group_ids=[pgid for pgid, _ in fuzzy_matches][:max_products],
                    match_stage="fuzzy",
                    confidence_score=avg_score / 100.0
                )

        # Stage 4: Token overlap fallback
        if query:
            token_matches = self._token_overlap_match(website_id, query)
            if token_matches:
                return MatchResult(
                    product_group_ids=token_matches,
                    match_stage="token_overlap",
                    confidence_score=0.5
                )

        return MatchResult(
            product_group_ids=[],
            match_stage="unmatched",
            confidence_score=0.0
        )

    def build_targeting_table(self, df_ads: pd.DataFrame) -> pd.DataFrame:
        """
        Build targeting table mapping ads to products.

        Args:
            df_ads: DataFrame with columns:
                - websiteId
                - platform
                - campaignId, adSetId, adId
                - destinationUrl (or destination_url)
                - name (ad name, optional)
                - collections (optional)

        Returns:
            DataFrame with columns:
                - websiteId, platform, campaignId, adSetId, adId
                - productGroupIds_targeted (list)
                - match_stage
                - confidence_score
        """
        self.logger.info(f"Building targeting table for {len(df_ads)} ads")

        # Normalize column names
        ads = df_ads.copy()
        if "destinationUrl" not in ads.columns and "destination_url" in ads.columns:
            ads["destinationUrl"] = ads["destination_url"]
        if "destinationUrl" not in ads.columns:
            ads["destinationUrl"] = ""
        if "name" not in ads.columns:
            ads["name"] = ""
        if "collections" not in ads.columns:
            ads["collections"] = ""

        # De-duplicate by unique signature to avoid repeated matching
        sig_cols = ["websiteId", "destinationUrl", "name", "collections"]
        for col in sig_cols:
            if col not in ads.columns:
                ads[col] = ""
            ads[col] = ads[col].apply(safe_str)

        signatures = ads[sig_cols].drop_duplicates().reset_index(drop=True)

        # Match each unique signature
        results = []
        for _, row in signatures.iterrows():
            match = self.match_ad(
                website_id=row["websiteId"],
                destination_url=row["destinationUrl"],
                ad_name=row["name"],
                collections=row["collections"]
            )
            results.append({
                "websiteId": row["websiteId"],
                "destinationUrl": row["destinationUrl"],
                "name": row["name"],
                "collections": row["collections"],
                "productGroupIds_targeted": match.product_group_ids,
                "match_stage": match.match_stage,
                "confidence_score": match.confidence_score,
            })

        sig_results = pd.DataFrame(results)

        # Merge back to all ad rows
        ads_with_match = ads.merge(sig_results, on=sig_cols, how="left")

        # Aggregate to ad level (unique campaignId/adSetId/adId)
        ad_keys = ["websiteId", "platform", "campaignId", "adSetId", "adId"]

        def union_lists(series):
            seen = set()
            result = []
            for lst in series:
                if isinstance(lst, list):
                    for item in lst:
                        if item not in seen:
                            seen.add(item)
                            result.append(item)
            return result

        # Priority order for match stages
        stage_priority = {
            "exact_url": 1,
            "exact_product": 2,
            "fuzzy": 3,
            "token_overlap": 4,
            "unmatched": 99
        }

        def best_stage(stages):
            stages = [safe_str(s) for s in stages if safe_str(s)]
            if not stages:
                return "unmatched"
            return min(stages, key=lambda s: stage_priority.get(s, 50))

        targeting = ads_with_match.groupby(ad_keys, as_index=False).agg({
            "productGroupIds_targeted": union_lists,
            "match_stage": best_stage,
            "confidence_score": "max",
        })

        self.logger.info(f"Targeting table built: {len(targeting)} unique ads")
        return targeting

    def flag_lead_products(
        self,
        targeting_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        product_id_col: str = "productGroupId"
    ) -> pd.DataFrame:
        """
        Add isLead flag to metrics DataFrame based on targeting.

        Args:
            targeting_df: Result from build_targeting_table()
            metrics_df: DataFrame with ad metrics at product level
            product_id_col: Column name containing product group ID

        Returns:
            metrics_df with added 'isLead' column (1 = lead, 0 = halo)
        """
        self.logger.info("Flagging lead products in metrics")

        result = metrics_df.copy()
        result["isLead"] = 0

        if targeting_df.empty or metrics_df.empty:
            return result

        # Explode targeting to get individual product mappings
        targeting_exp = targeting_df.copy()
        targeting_exp = targeting_exp[
            targeting_exp["productGroupIds_targeted"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        ]

        if targeting_exp.empty:
            return result

        targeting_exp = targeting_exp.explode("productGroupIds_targeted")
        targeting_exp = targeting_exp.rename(columns={"productGroupIds_targeted": product_id_col})

        # Get unique (ad, product) pairs that are leads
        lead_keys = targeting_exp[
            ["campaignId", "adSetId", "adId", product_id_col]
        ].drop_duplicates()

        # Merge to identify leads
        merged = result.merge(
            lead_keys,
            on=["campaignId", "adSetId", "adId", product_id_col],
            how="left",
            indicator=True
        )

        result["isLead"] = (merged["_merge"] == "both").astype(int)

        lead_count = result["isLead"].sum()
        self.logger.info(f"Flagged {lead_count} lead product rows out of {len(result)} total")

        return result

    def run(
        self,
        df_ads: pd.DataFrame,
        df_products: pd.DataFrame,
        df_metrics: Optional[pd.DataFrame] = None,
    ) -> TargetingResult:
        """
        Run the full ad-to-product matching pipeline.

        Args:
            df_ads: Ads with destination URLs
            df_products: Product catalog
            df_metrics: Optional product-level metrics to flag as lead/halo

        Returns:
            TargetingResult with targeting table and flagged metrics
        """
        self.logger.info("Starting ad-product matching pipeline")

        # Build product index
        self.build_product_index(df_products)

        # Build targeting table
        targeting_df = self.build_targeting_table(df_ads)

        # Flag lead products in metrics
        metrics_flagged = None
        if df_metrics is not None and not df_metrics.empty:
            metrics_flagged = self.flag_lead_products(targeting_df, df_metrics)

        return TargetingResult(
            targeting_df=targeting_df,
            metrics_df=metrics_flagged if metrics_flagged is not None else pd.DataFrame(),
        )
