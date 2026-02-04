# Usage Guide: Campaign Product Impression AI Assist

This guide explains how to create the data and use the Streamlit dashboard (`app.py`) for analyzing campaign and product performance.

---

## Table of Contents

1. [Overview](#overview)
2. [Creating the Data](#creating-the-data)
   - [Option A: Using the Simple Pipeline](#option-a-using-the-simple-pipeline-recommended)
   - [Option B: Using the Metric-Only Pipeline](#option-b-using-the-metric-only-pipeline)
   - [Option C: Using the Full Product-Matching Pipeline](#option-c-using-the-full-product-matching-pipeline)
   - [Option D: Preparing Data Manually](#option-d-preparing-data-manually)
3. [Using the Dashboard (app.py)](#using-the-dashboard-apppy)
4. [Data Schema Reference](#data-schema-reference)
5. [Troubleshooting](#troubleshooting)

---

## Overview

This platform helps you:
- **Match ads to products**: Identify which products each ad is targeting
- **Measure spillover effects**: Distinguish between "lead" products (directly targeted) and "halo" products (spillover sales)
- **Analyze performance**: View efficiency, scalability, and actionable recommendations

---

## Creating the Data

You have four options for creating the data that `app.py` requires.

### Option A: Using the Simple Pipeline (Recommended)

The pipeline automatically matches ads to products and computes allocated metrics.

#### Step 1: Prepare Input Files

You need three input files in Parquet or CSV format:

**1. `ads.parquet`** - Your ad data:
| Column | Required | Description |
|--------|----------|-------------|
| `websiteId` | Yes | Website identifier |
| `platform` | Yes | google, meta, tiktok, pinterest |
| `campaignId` | Yes | Campaign identifier |
| `adSetId` | Yes | Ad set identifier |
| `adId` | Yes | Ad identifier |
| `destinationUrl` | Yes | Ad destination URL |
| `name` | No | Ad name/title (helps with matching) |

**2. `products.parquet`** - Your product catalog:
| Column | Required | Description |
|--------|----------|-------------|
| `websiteId` | Yes | Website identifier |
| `productGroupId` | Yes | Product group identifier |
| `name` | Yes | Product name |
| `url` | No | Product page URL (improves matching) |

**3. `metrics.parquet`** - Performance metrics:
| Column | Required | Description |
|--------|----------|-------------|
| `campaignId` | Yes | Campaign identifier |
| `adSetId` | Yes | Ad set identifier |
| `adId` | Yes | Ad identifier |
| `productGroupId` or `productId` | Yes | Product being attributed |
| `spend` | Yes | Ad spend |
| `impressions` | Yes | Ad impressions |
| `date` | No | Date of the metrics |
| `conversions` | No | Conversion count |
| `grossProfit` | No | Revenue/profit |

#### Step 2: Run the Pipeline

**From command line:**
```bash
# Place your files in a ./data folder, then run:
python -m simple.run_pipeline --from-parquet ./data --output-dir ./results --format both
```

**From Python:**
```python
from simple import PipelineConfig, AdProductMatcher, compute_all_metrics
import pandas as pd

# 1. Configure
config = PipelineConfig(
    website_id="your-website-id",
    fuzzy_threshold=85,  # Higher = stricter matching
)

# 2. Load your data
ads_df = pd.read_parquet("data/ads.parquet")
products_df = pd.read_parquet("data/products.parquet")
metrics_df = pd.read_parquet("data/metrics.parquet")

# 3. Match ads to products
matcher = AdProductMatcher(config)
matcher.build_product_index(products_df)
targeting = matcher.build_targeting_table(ads_df)

# 4. Flag lead products in metrics
metrics_with_flags = matcher.flag_lead_products(targeting, metrics_df)

# 5. Compute allocated metrics
results = compute_all_metrics(metrics_with_flags)

# 6. Save for dashboard
results.sku_allocation.to_parquet("results/sku_allocation.parquet")
results.sku_allocation.to_csv("results/sku_allocation.csv", index=False)
```

#### Step 3: Output Files

The pipeline generates `sku_allocation.csv/parquet` which is ready for `app.py`.

---

### Option B: Using the Metric-Only Pipeline

If you already have a **coalesced product table** (one row per product per ad per day),
use the metric pipeline to allocate ad totals and produce the same `sku_allocation` output
that `app.py` expects.

#### Step 1: Prepare a Coalesced Product Table

This table must include:
- `date` or `timestamp`
- `platform`, `campaignId`, `adSetId`, `adId`
- `productId` (or `productId_`)

Optional columns such as `productGroupId`, `productGroupName`, `productName`, `spend`,
`impressions`, `clicks`, `conversions`, `grossProfit`, and `isLead` are supported and will
be normalized if present.

#### Step 2: Run the Metric Pipeline

```bash
python getting-metric.py --input ./data/coalesced_df.parquet --output-dir ./results --formats csv,parquet
```

By default, `getting-metric.py` uses `conversionsValue` as the profit column. Override with:

```bash
python getting-metric.py --input ./data/coalesced_df.parquet --profit-col grossProfit
```

The output `results/sku_allocation.csv/parquet` can be uploaded directly into `app.py`.

---

### Option C: Using the Full Product-Matching Pipeline

If you want end-to-end attribution (ads → product groups → metrics), run the full
product-matching workflow.

#### Step 1: Ensure Input Parquet Files Exist

`product-matching.py` expects these files in the repo root:

- `websites.parquet`
- `df_ads_ori.parquet`
- `df_prod.parquet`
- `df_prod_group.parquet`
- `df_metrics.parquet`
- `df_metrics_by_country.parquet`
- `df_metrics_by_products.parquet`

#### Step 2: Run the Workflow

```bash
python product-matching.py
```

Outputs are written to `results/`, including:
- `results/final_targeting.parquet`
- `results/coalesced_df_with_flags.parquet`
- `results/metric-output/sku_allocation.csv/parquet`

Use `results/metric-output/sku_allocation.*` as the input for `app.py`.

---

### Option D: Preparing Data Manually

If you already have your own attribution data, you can create a CSV/Parquet file directly.

#### Required Columns for app.py

Your file **must** contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `date` | date/string | Date of the record |
| `productGroupName` | string | Product group name |
| `productName` | string | Product name |
| `platform` | string | Ad platform (google, meta, etc.) |
| `campaignId` | string | Campaign identifier |
| `campaignName` | string | Campaign name |
| `adSetId` | string | Ad set identifier |
| `adSetName` | string | Ad set name |
| `adId` | string | Ad identifier |
| `adName` | string | Ad name |
| `isLead` | 0/1 | 1 = lead product, 0 = halo/spillover |
| `gross_profit_fair` | numeric | Gross profit allocated to all products |
| `spend_fair` | numeric | Spend allocated to all products |
| `impressions_fair` | numeric | Impressions allocated to all products |
| `gross_profit_lead_only` | numeric | Gross profit for lead products only |
| `spend_lead_only` | numeric | Spend for lead products only |
| `impressions_lead_only` | numeric | Impressions for lead products only |

#### Optional Column

| Column | Type | Description |
|--------|------|-------------|
| `match_stage` | string | Match confidence: `exact_url`, `exact_product`, `fuzzy`, `token_overlap`, `unmatched` |
| `productId` | string | Optional SKU/product identifier |

#### Example Data

```csv
date,productGroupName,productName,platform,campaignId,campaignName,adSetId,adSetName,adId,adName,isLead,gross_profit_fair,spend_fair,impressions_fair,gross_profit_lead_only,spend_lead_only,impressions_lead_only,match_stage
2025-01-15,Shoes,Running Shoe Pro,google,camp_001,Brand Campaign,adset_001,Athletic Audience,ad_001,Running Shoe Ad,1,500.00,100.00,5000,500.00,100.00,5000,exact_url
2025-01-15,Shoes,Casual Sneaker,google,camp_001,Brand Campaign,adset_001,Athletic Audience,ad_001,Running Shoe Ad,0,150.00,30.00,1500,0.00,0.00,0,token_overlap
```

---

## Using the Dashboard (app.py)

### Starting the Dashboard

```bash
streamlit run app.py
```

This opens a browser window with the dashboard.

### Uploading Data

1. In the **sidebar**, click "Browse files" under "Upload CSV / Parquet"
2. Select your `sku_allocation.csv` or `sku_allocation.parquet` file

### Dashboard Sections

#### Sidebar Controls

| Control | Description |
|---------|-------------|
| **Scoring thresholds** | Set thresholds for action recommendations |
| **Min impressions** | Minimum impressions to trust a row (default: 2000) |
| **Scale if GP/Spend >=** | Efficiency threshold to recommend scaling |
| **View options** | Choose Campaign-first, ProductGroup-first, or Campaign-only |
| **Metric scope** | Total (fair), Lead-only, or Halo metrics |
| **Page filters** | Date range, platform, campaign, product group filters |
| **Match stage filter** | Filter by match confidence level |

#### 1. Summary Tiles

Shows high-level KPIs:
- **Gross Profit (Total)**: Sum of all gross profit
- **Spend (Total)**: Sum of all ad spend
- **GP / Spend**: Efficiency ratio (higher is better)
- **GP per 1k Impr**: Scalability metric
- **Spillover Share**: % of profit from halo (non-lead) products

#### 2. Actionable Leaderboard

A table showing campaigns/products with action recommendations:
- **1. Collect more data**: Insufficient impressions
- **2. Scale**: High efficiency AND high scale
- **3. Keep / Monitor**: Good performance
- **4. Improve targeting / creative**: High efficiency but low scale
- **5. Fix economics**: Low efficiency but high scale

#### 3. Quadrant Chart (Scatter)

Visualizes efficiency vs scalability:
- X-axis: GP/Spend (efficiency)
- Y-axis: Impressions or GP per 1k (scalability)
- Bubble size: Total spend

#### 4. Lead vs Halo (Stacked Bar)

Shows the breakdown of gross profit/spend between:
- **Lead products**: Directly targeted by the ad
- **Halo products**: Spillover/discovery sales

#### 5. Spillover Insights

Scatter plot and tables showing:
- Campaigns with highest spillover (discovery-like behavior)
- Campaigns with lowest spillover (lead-product focused)

#### 6. Trend Analysis

Time series charts to monitor KPI changes over time.

#### 7. Waste Detection

Identifies where spend is losing money (GP/Spend < 1.0).

---

## Data Schema Reference

### Understanding Lead vs Halo

| Type | Description | isLead | Match Stage |
|------|-------------|--------|-------------|
| **Lead** | Product directly targeted by ad | 1 | exact_url, exact_product, fuzzy |
| **Halo** | Spillover sale from ad exposure | 0 | token_overlap, unmatched |

### Metric Allocations

| Metric Type | Description |
|-------------|-------------|
| `*_fair` | Metrics allocated proportionally to ALL products attributed to an ad |
| `*_lead_only` | Metrics allocated ONLY to lead products (halo gets 0) |

### Match Stages (Confidence)

| Stage | Confidence | Description |
|-------|------------|-------------|
| `exact_url` | 100% | Ad URL exactly matches product URL |
| `exact_product` | 95% | Product slug found in ad URL |
| `fuzzy` | 85-99% | Ad name fuzzy-matches product name |
| `token_overlap` | 50% | Keywords overlap (fallback) |
| `unmatched` | N/A | No match found (halo product) |

---

## Troubleshooting

### "Missing required columns" error

Ensure your file has ALL required columns listed in [Required Columns](#required-columns-for-apppy).

### No products matched in pipeline

- Lower `fuzzy_threshold` (try 70-80)
- Check that product URLs/names are populated
- Verify `websiteId` matches between ads and products

### Product-matching workflow cannot find parquet files

- Confirm the parquet files listed in [Option C](#option-c-using-the-full-product-matching-pipeline) exist
- Run the notebook or upstream export that generates the parquet files

### Dashboard shows "No rows after filters"

- Check your date range filter
- Ensure the selected platforms exist in your data
- Remove strict filters temporarily

### Empty charts

- Increase `Top N rows` slider
- Uncheck "Hide rows with impressions < min"
- Clear action filters

### Match stage column missing

The dashboard will automatically set `match_stage` to "unmatched" if missing. For better insights, run the simple pipeline which generates this column.

---

## Quick Start Checklist

1. [ ] Prepare your input data (ads, products, metrics)
2. [ ] Run the pipeline: `python -m simple.run_pipeline --from-parquet ./data`
3. [ ] Start dashboard: `streamlit run app.py`
4. [ ] Upload `sku_allocation.csv` or `.parquet`
5. [ ] Adjust filters and explore insights
