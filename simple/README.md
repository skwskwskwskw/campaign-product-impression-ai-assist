# Simple Ad-Product Attribution Pipeline

A simplified, easy-to-understand pipeline for identifying which products each ad is targeting
and measuring spillover effects using product-level metrics.

## Overview

For each **campaignId/adSetId/adId**, this pipeline determines:
1. **Lead Products**: Products explicitly targeted by the ad (via URL, name matching)
2. **Halo/Spillover Products**: Other products that sell as a result of the ad

### Key Metrics

| Metric | Description |
|--------|-------------|
| `spend_fair` | Ad spend allocated proportionally to all attributed products |
| `spend_lead_only` | Ad spend allocated only to lead products |
| `gross_profit_fair` | Gross profit from all products attributed to the ad |
| `gross_profit_lead_only` | Gross profit from lead products only |
| `spillover_share` | % of gross profit from halo (non-lead) products |

## Quick Start

### From Command Line

```bash
# From parquet files:
python -m simple.run_pipeline --from-parquet ./data --output-dir ./results --format both

# From ClickHouse:
python -m simple.run_pipeline --website-id YOUR_WEBSITE_ID \
    --start-date 2025-01-01 --end-date 2025-01-31
```

### From Python

```python
from simple import PipelineConfig, AdProductMatcher, compute_all_metrics, compute_summary_kpis
import pandas as pd

# 1. Configure
config = PipelineConfig(
    website_id="your-website-id",
    fuzzy_threshold=85,  # Higher = stricter matching
)

# 2. Load your data
ads_df = pd.read_parquet("ads.parquet")
products_df = pd.read_parquet("products.parquet")
metrics_df = pd.read_parquet("metrics.parquet")

# 3. Match ads to products
matcher = AdProductMatcher(config)
matcher.build_product_index(products_df)
targeting = matcher.build_targeting_table(ads_df)

# 4. Flag lead products in metrics
metrics_with_flags = matcher.flag_lead_products(targeting, metrics_df)

# 5. Compute allocated metrics
results = compute_all_metrics(metrics_with_flags)
summary = compute_summary_kpis(results.sku_allocation)

# 6. Access results
print(results.sku_allocation.head())  # Product-level allocations
print(results.ad_data.head())         # Ad-level totals
print(summary.head())                 # Campaign KPIs
```

## Input Data Requirements

### Ads DataFrame (`ads_df`)
| Column | Required | Description |
|--------|----------|-------------|
| `websiteId` | Yes | Website identifier |
| `platform` | Yes | google, meta, tiktok, pinterest |
| `campaignId` | Yes | Campaign identifier |
| `adSetId` | Yes | Ad set identifier |
| `adId` | Yes | Ad identifier |
| `destinationUrl` | Yes | Ad destination URL |
| `name` | No | Ad name/title (helps matching) |

### Products DataFrame (`products_df`)
| Column | Required | Description |
|--------|----------|-------------|
| `websiteId` | Yes | Website identifier |
| `productGroupId` | Yes | Product group identifier |
| `name` | Yes | Product name |
| `url` | No | Product page URL |

### Metrics DataFrame (`metrics_df`)
| Column | Required | Description |
|--------|----------|-------------|
| `campaignId` | Yes | Campaign identifier |
| `adSetId` | Yes | Ad set identifier |
| `adId` | Yes | Ad identifier |
| `productGroupId` or `productId` | Yes | Product being attributed |
| `spend` | Yes | Ad spend |
| `impressions` | Yes | Ad impressions |
| `conversions` | No | Conversion count |
| `grossProfit` | No | Revenue/profit (alias: conversionsValue) |

## Output Tables

### targeting.csv/parquet
Ad-to-product mappings with match confidence.

```
campaignId, adSetId, adId, productGroupIds_targeted, match_stage, confidence_score
```

### sku_allocation.csv/parquet
Product-level metrics with both fair and lead-only allocations.

```
date, productId, campaignId, adSetId, adId, isLead,
spend_fair, impressions_fair, gross_profit_fair,
spend_lead_only, impressions_lead_only, gross_profit_lead_only
```

### sku_performance.csv/parquet
Aggregated product performance across all ads.

```
date, productId, sku_spend_fair, sku_impressions_fair, sku_clicks_fair,
sku_gross_profit_fair, sku_spend_lead_only, sku_impressions_lead_only,
sku_clicks_lead_only, sku_gross_profit_lead_only
```

### campaign_summary.csv/parquet
Campaign-level KPIs including spillover analysis.

```
campaignId, campaignName, gp_total, spend_total,
gp_lead, gp_halo, spillover_share, gp_per_spend_total
```

## Matching Logic

The pipeline matches ads to products in stages (in order of confidence):

1. **Exact URL Match** (confidence: 1.0)
   - Ad destination URL exactly matches product URL
   - Example: `mystore.com/products/blue-shirt` → Product "Blue Shirt"

2. **Product Slug Match** (confidence: 0.95)
   - Product slug in URL matches product
   - Example: `/products/blue-shirt` → Products containing "blue-shirt"

3. **Fuzzy Name Match** (confidence: 0.85-0.99)
   - Ad name fuzzy-matches product name
   - Uses rapidfuzz library with configurable threshold

4. **Token Overlap** (confidence: 0.5)
   - Keywords in ad match keywords in product
   - Fallback when other methods fail

## Configuration Options

Edit `config.py` or pass to `PipelineConfig`:

```python
PipelineConfig(
    # Matching
    fuzzy_threshold=85,      # Min fuzzy score (0-100)
    fuzzy_limit=30,          # Max fuzzy candidates per query
    min_token_length=3,      # Min chars for token matching
    max_products_per_ad=50,  # Cap products per ad

    # Data
    profit_column="conversionsValue",  # Revenue column name
    weight_column="conversions",       # Allocation weight column

    # Output
    output_dir="./results",
    output_formats=["csv", "parquet"],

    # ClickHouse (optional)
    start_date="2025-10-01",
    end_date="2025-12-31",
    use_aws_secrets=True,
)
```

## Customization Points

### Change Matching Logic
Edit `simple/ad_product_matcher.py`:
- `match_ad()` - Main matching function
- `_fuzzy_match()` - Fuzzy matching algorithm
- `build_product_index()` - Index construction

### Change Metrics Calculation
Edit `simple/metrics.py`:
- `compute_all_metrics()` - Main computation
- `compute_summary_kpis()` - Campaign-level KPIs

### Add New Data Sources
Edit `simple/run_pipeline.py`:
- `load_from_parquet()` - Parquet loading
- `load_from_clickhouse()` - Database loading

## File Structure

```
simple/
├── __init__.py           # Package exports
├── config.py             # Configuration settings
├── ad_product_matcher.py # Core matching logic
├── metrics.py            # Metrics computation
├── run_pipeline.py       # CLI entry point
└── README.md             # This file
```

## Troubleshooting

### No products matched
- Lower `fuzzy_threshold` (try 70-80)
- Check that product URLs/names are populated
- Verify `websiteId` matches between ads and products

### Too many products matched
- Raise `fuzzy_threshold` (try 90-95)
- Lower `max_products_per_ad`

### Missing metrics
- Ensure `productGroupId` or `productId` exists in metrics
- Check date range includes your data

## Dependencies

- pandas >= 1.0
- numpy >= 1.19
- rapidfuzz >= 2.0 (optional, falls back to difflib)
- clickhouse-connect >= 0.7 (optional, for ClickHouse ingestion)
- boto3 >= 1.28 (optional, for AWS Secrets Manager credentials)
