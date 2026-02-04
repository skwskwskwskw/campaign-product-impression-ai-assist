"""
Ad targeting and product matching functions with validation and error handling.
"""

import re
import pandas as pd
import logging
from urllib.parse import urlparse, unquote
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Any


# Optional: fast fuzzy (recommended)
try:
    from rapidfuzz import fuzz
    from rapidfuzz import process as rf_process
    HAS_RAPIDFUZZ = True
except Exception:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False


# =========================
# Config
# =========================
MIN_TOKEN_LEN = 3
FUZZY_THRESHOLD = 85     # you can raise to 90 if you want stricter fuzzy
FUZZY_LIMIT = 30         # top-N fuzzy hits kept per query
KEEP_TOP_N = 50          # cap list lengths to avoid huge lists

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


# =========================
# Safe / normalize helpers
# =========================
def safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def safe_lower(x) -> str:
    return safe_str(x).lower()

def normalize_for_match(s: str) -> str:
    s = safe_str(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def tokenize(s: str) -> list[str]:
    s = safe_str(s).lower()
    if not s:
        return []
    toks = [re.sub(r"[^a-z0-9]+", "", t) for t in WORD_RE.findall(s)]
    return [t for t in toks if t and len(t) >= MIN_TOKEN_LEN]

def canonical_url(u: str) -> str:
    """
    Canonical compare: host + path (drop query/fragment), decode %xx, strip trailing slash.
    """
    u = safe_str(u)
    if not u:
        return ""
    u = unquote(u)
    if "://" not in u and u.startswith("www."):
        u = "https://" + u
    p = urlparse(u)
    host = safe_str(p.netloc).lower()
    path = safe_str(p.path)
    path = re.sub(r"/{2,}", "/", path).rstrip("/")
    return f"{host}{path}" if (host or path) else ""


# =========================
# Build product indices
# =========================
def build_product_indices(df_prod_group: pd.DataFrame):
    """
    Build product indices with validation and error handling.
    
    Args:
        df_prod_group: Product group DataFrame
        
    Returns:
        Dictionary containing product indices
    """
    logging.info(f"Building product indices for df_prod_group with shape: {df_prod_group.shape}")
    
    pg = df_prod_group.copy()

    # Required columns
    required = ["websiteId", "productGroupId", "url_beautified_decoded", "products_", "name"]
    missing = [c for c in required if c not in pg.columns]
    if missing:
        raise ValueError(f"df_prod_group missing columns: {missing}")

    pg["websiteId"] = pg["websiteId"].map(safe_str)
    pg["productGroupId"] = pg["productGroupId"].map(safe_str)

    pg["url_canon"] = pg["url_beautified_decoded"].map(canonical_url)
    pg["products_norm"] = pg["products_"].map(safe_lower)
    pg["name_norm"] = pg["name"].map(safe_lower)

    # Exact maps
    url_map = (
        pg[pg["url_canon"] != ""]
        .groupby(["websiteId", "url_canon"])["productGroupId"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )
    prod_map = (
        pg[pg["products_norm"] != ""]
        .groupby(["websiteId", "products_norm"])["productGroupId"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )

    # Fuzzy corpora (per website)
    # We'll build two corpora:
    # - products_norm corpus
    # - name_norm corpus
    fuzzy_corp_prod = {}
    fuzzy_corp_name = {}

    if HAS_RAPIDFUZZ:
        tmp = pg[["websiteId", "productGroupId", "products_norm", "name_norm"]].drop_duplicates()
        for wid, g in tmp.groupby("websiteId"):
            # products corpus
            prod_strings = []
            prod_pgids = []
            # name corpus
            name_strings = []
            name_pgids = []

            for _, r in g.iterrows():
                pgid = safe_str(r["productGroupId"])
                pr = safe_str(r["products_norm"])
                nm = safe_str(r["name_norm"])
                if pr:
                    prod_strings.append(pr)
                    prod_pgids.append(pgid)
                if nm:
                    name_strings.append(nm)
                    name_pgids.append(pgid)

            fuzzy_corp_prod[wid] = (prod_strings, prod_pgids)
            fuzzy_corp_name[wid] = (name_strings, name_pgids)

    # Token overlap index (final fallback)
    # index token -> set(productGroupId) for (products_norm + name_norm)
    token_index = defaultdict(set)
    pg_tokens = pg[["websiteId", "productGroupId", "products_norm", "name_norm"]].drop_duplicates().copy()
    pg_tokens["tokset"] = pg_tokens.apply(
        lambda r: set(tokenize(r["products_norm"])) | set(tokenize(r["name_norm"])),
        axis=1
    )
    for _, r in pg_tokens.iterrows():
        wid = r["websiteId"]
        pgid = r["productGroupId"]
        for t in r["tokset"]:
            token_index[(wid, t)].add(pgid)

    logging.info(f"Built product indices: {len(url_map)} URL mappings, {len(prod_map)} product mappings")
    
    return {
        "pg": pg,
        "url_map": url_map,
        "prod_map": prod_map,
        "fuzzy_corp_prod": fuzzy_corp_prod,
        "fuzzy_corp_name": fuzzy_corp_name,
        "token_index": token_index,
    }


# =========================
# Fuzzy utilities
# =========================
def _fallback_ratio(a: str, b: str) -> int:
    a = safe_str(a); b = safe_str(b)
    if not a or not b:
        return 0
    return int(100 * SequenceMatcher(None, a, b).ratio())

def fuzzy_match_pgids(wid: str, query: str, corpus_dict: dict, threshold: int, limit: int):
    """
    Returns list of (pgid, score) best score per pgid.
    """
    query = safe_lower(query)
    if not query:
        return []
    if HAS_RAPIDFUZZ and wid in corpus_dict:
        strings, pgids = corpus_dict[wid]
        if not strings:
            return []
        try:
            res = rf_process.extract(
                query,
                strings,
                scorer=fuzz.ratio,
                score_cutoff=threshold,
                limit=limit
            )
            best = {}
            for _m, sc, idx in res:
                pgid = pgids[idx]
                sc = int(sc)
                if pgid not in best or sc > best[pgid]:
                    best[pgid] = sc
            return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        except Exception as e:
            logging.warning(f"Fuzzy matching failed for website {wid}: {e}")
            return []

    # Slow fallback if no rapidfuzz
    # (only used if rapidfuzz not installed)
    # NOTE: you can remove this fallback if you always have rapidfuzz.
    # This fallback requires caller to have passed a corpus list instead; kept simple:
    return []


# =========================
# Match one ad signature (strict staged)
# =========================
def match_one_signature(row, idx, cache):
    """
    row columns expected:
      websiteId, dest_canon, products_norm, ad_name, collections_
    """
    wid = row["websiteId"]
    dest_canon = row["dest_canon"]
    prod_norm = row["products_norm"]
    ad_name = row["ad_name"]
    collections_ = row["collections_"]

    # Build query for fuzzy (as requested):
    # ad_name / collections_ / products_
    query_parts = [ad_name, collections_, prod_norm]
    query = " ".join([safe_str(x) for x in query_parts if safe_str(x)])

    # --------------------------
    # Stage 1: exact URL
    # --------------------------
    if dest_canon:
        pgids = idx["url_map"].get((wid, dest_canon), [])
        if pgids:
            return {
                "productGroupIds_targeted": pgids[:KEEP_TOP_N],
                "match_stage": "exact_url",
            }

    # --------------------------
    # Stage 1b: exact products_
    # --------------------------
    if prod_norm:
        pgids = idx["prod_map"].get((wid, prod_norm), [])
        if pgids:
            return {
                "productGroupIds_targeted": pgids[:KEEP_TOP_N],
                "match_stage": "exact_products_",
            }

    # --------------------------
    # Stage 2: fuzzy (only if Stage 1 failed)
    #   2a) query vs prod_group.products_
    #   2b) query vs prod_group.name
    # --------------------------
    if FUZZY_THRESHOLD is not None and query:
        # cache fuzzy results per (wid, query, target)
        k1 = ("fuzzy_prod", wid, query)
        if k1 not in cache:
            cache[k1] = fuzzy_match_pgids(wid, query, idx["fuzzy_corp_prod"], FUZZY_THRESHOLD, FUZZY_LIMIT)
        hits_prod = cache[k1]

        if hits_prod:
            pgids = [pgid for pgid, _ in hits_prod][:KEEP_TOP_N]
            return {
                "productGroupIds_targeted": pgids,
                "match_stage": "fuzzy_vs_products_",
            }

        k2 = ("fuzzy_name", wid, query)
        if k2 not in cache:
            cache[k2] = fuzzy_match_pgids(wid, query, idx["fuzzy_corp_name"], FUZZY_THRESHOLD, FUZZY_LIMIT)
        hits_name = cache[k2]

        if hits_name:
            pgids = [pgid for pgid, _ in hits_name][:KEEP_TOP_N]
            return {
                "productGroupIds_targeted": pgids,
                "match_stage": "fuzzy_vs_name",
            }

    # --------------------------
    # Stage 3: token overlap fallback (>= 3 chars tokens)
    # (adapted from old "collections matching" idea)
    # --------------------------
    q_tokens = tokenize(query)
    if q_tokens:
        scores = defaultdict(int)
        for t in set(q_tokens):
            for pgid in idx["token_index"].get((wid, t), set()):
                scores[pgid] += 1

        if scores:
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            pgids = [pgid for pgid, _ in ranked][:KEEP_TOP_N]
            return {
                "productGroupIds_targeted": pgids,
                "match_stage": "token_overlap_fallback",
            }

    return {
        "productGroupIds_targeted": [],
        "match_stage": "unmatched",
    }


# =========================
# Main runner
# =========================
def build_targeting(
    df_ads_enhanced: pd.DataFrame,
    df_prod_group: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build ad targeting with validation and error handling.
    
    Args:
        df_ads_enhanced: Enhanced ads DataFrame
        df_prod_group: Product group DataFrame
        
    Returns:
        Tuple of (final targeting DataFrame, debug rows DataFrame)
    """
    logging.info(f"Starting ad targeting: ads={df_ads_enhanced.shape}, prod_group={df_prod_group.shape}")
    
    ads = df_ads_enhanced.copy()

    required_ads = [
        "websiteId","platform","campaignId","adSetId","adId",
        "destinationUrl_beautified_decoded","products_","ad_name"
    ]
    missing = [c for c in required_ads if c not in ads.columns]
    if missing:
        raise ValueError(f"df_ads_enhanced missing columns: {missing}")

    # Optional collections_
    if "collections_" not in ads.columns:
        ads["collections_"] = ""

    # Normalize ads fields
    for c in required_ads + ["collections_"]:
        ads[c] = ads[c].map(safe_str)

    ads["dest_canon"] = ads["destinationUrl_beautified_decoded"].map(canonical_url)
    ads["products_norm"] = ads["products_"].map(safe_lower)

    # Build product indices once
    idx = build_product_indices(df_prod_group)

    # ---- HUGE speed win: match unique signatures only ----
    sig_cols = ["websiteId", "dest_canon", "products_norm", "ad_name", "collections_"]
    sig = ads[sig_cols].drop_duplicates().reset_index(drop=True)

    cache = {}
    matched = sig.apply(lambda r: match_one_signature(r, idx, cache), axis=1)
    matched_df = pd.DataFrame(matched.tolist())

    sig_out = pd.concat([sig, matched_df], axis=1)

    # Merge back to all ads rows
    ads2 = ads.merge(sig_out, on=sig_cols, how="left")

    # Aggregate to adId level
    keys = ["websiteId", "platform", "campaignId", "adSetId", "adId"]

    def union_lists(series):
        seen = set()
        out = []
        for lst in series:
            if isinstance(lst, list):
                for x in lst:
                    if x not in seen:
                        seen.add(x)
                        out.append(x)
        return out

    # pick "best" stage by priority order (exact_url > exact_products_ > fuzzy_prod > fuzzy_name > token > unmatched)
    stage_rank = {
        "exact_url": 1,
        "exact_products_": 2,
        "fuzzy_vs_products_": 3,
        "fuzzy_vs_name": 4,
        "token_overlap_fallback": 5,
        "unmatched": 99
    }

    def best_stage(stages):
        stages = [safe_str(s) for s in stages if safe_str(s)]
        if not stages:
            return "unmatched"
        return sorted(stages, key=lambda s: stage_rank.get(s, 50))[0]

    final = (
        ads2.groupby(keys, as_index=False)
        .agg({
            "productGroupIds_targeted": union_lists,
            "match_stage": best_stage,
        })
    )

    logging.info(f"Ad targeting completed: final={final.shape}, debug={ads2.shape}")
    
    return final, ads2