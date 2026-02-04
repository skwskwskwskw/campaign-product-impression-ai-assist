"""
Simple Ad-Product Attribution Pipeline

This package provides a simplified, easy-to-edit implementation for:
- Matching ads to products (identifying lead vs halo products)
- Allocating ad metrics (spend, impressions) to products
- Computing spillover/halo effects

Main components:
- config.py: Configuration settings
- ad_product_matcher.py: Core matching logic
- metrics.py: Metrics computation
- run_pipeline.py: CLI entry point

Quick start:
    from simple.config import PipelineConfig
    from simple.ad_product_matcher import AdProductMatcher
    from simple.metrics import compute_all_metrics

    # Configure
    config = PipelineConfig(website_id="your-id")

    # Match ads to products
    matcher = AdProductMatcher(config)
    matcher.build_product_index(products_df)
    targeting = matcher.build_targeting_table(ads_df)

    # Flag lead products and compute metrics
    metrics_flagged = matcher.flag_lead_products(targeting, metrics_df)
    results = compute_all_metrics(metrics_flagged)

    # Access results
    results.sku_allocation   # Product-level metrics
    results.campaign_summary # Campaign KPIs
"""

from .config import PipelineConfig, default_config, local_config
from .ad_product_matcher import AdProductMatcher, MatchResult, TargetingResult
from .metrics import (
    compute_all_metrics,
    compute_summary_kpis,
    write_outputs,
    MetricsResult,
)

__all__ = [
    # Config
    "PipelineConfig",
    "default_config",
    "local_config",
    # Matcher
    "AdProductMatcher",
    "MatchResult",
    "TargetingResult",
    # Metrics
    "compute_all_metrics",
    "compute_summary_kpis",
    "write_outputs",
    "MetricsResult",
]

__version__ = "1.0.0"
