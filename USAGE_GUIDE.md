# Usage Guide: Campaign Product Impression AI Assist

This guide provides step-by-step instructions for preparing data and using the Streamlit dashboard (`app.py`) for analyzing campaign and product performance.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Pipeline Options](#pipeline-options)
   - [Option A: Simple Pipeline (Recommended)](#option-a-simple-pipeline-recommended)
   - [Option B: Metric-Only Pipeline](#option-b-metric-only-pipeline)
   - [Option C: Full Product-Matching Pipeline](#option-c-full-product-matching-pipeline)
   - [Option D: Manual Data Preparation](#option-d-manual-data-preparation)
4. [Configurable Parameters Reference](#configurable-parameters-reference)
5. [Using the Dashboard](#using-the-dashboard-apppy)
6. [Data Schema Reference](#data-schema-reference)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This platform helps you:
- **Match ads to products**: Identify which products each ad is targeting
- **Measure spillover effects**: Distinguish between "lead" products (directly targeted) and "halo" products (spillover sales)
- **Analyze performance**: View efficiency, scalability, and actionable recommendations

### Data Flow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ads.parquet   │     │ products.parquet│     │ metrics.parquet │
│   (Ad data)     │     │ (Product catalog)│    │ (Performance)   │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Pipeline Processing  │
                    │  (Matching + Metrics)  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  sku_allocation.parquet│
                    │  (Dashboard-ready)     │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      app.py Dashboard  │
                    │   (Visualization)      │
                    └────────────────────────┘
```

---

## Quick Start

```bash
# 1. Place your input files in ./data folder
mkdir -p data
# Copy ads.parquet, products.parquet, metrics.parquet to ./data

# 2. Run the pipeline
python -m simple.run_pipeline --from-parquet ./data --output-dir ./results --format both

# 3. Start the dashboard
streamlit run app.py

# 4. Upload results/sku_allocation.csv or .parquet in the dashboard
```

---

## Pipeline Options

Choose the pipeline that matches your data situation:

| Option | Use When | Input Required |
|--------|----------|----------------|
| **A: Simple Pipeline** | You have separate ads, products, metrics files | 3 files: ads, products, metrics |
| **B: Metric-Only** | You already have product-level metrics with ad mapping | 1 file: coalesced metrics |
| **C: Full Pipeline** | You need end-to-end processing from raw data | 7 parquet files |
| **D: Manual** | You want to create the dashboard file yourself | 1 file: sku_allocation |

---

## Option A: Simple Pipeline (Recommended)

The simple pipeline is the easiest way to get started. It automatically matches ads to products and computes allocated metrics.

### Step 1: Prepare Input Files

Create three files in your data directory:

#### 1.1 `ads.parquet` or `ads.csv`

| Column | Required | Type | Description |
|--------|----------|------|-------------|
| `websiteId` | **Yes** | string | Website identifier |
| `platform` | **Yes** | string | `google`, `meta`, `tiktok`, `pinterest` |
| `campaignId` | **Yes** | string | Campaign identifier |
| `adSetId` | **Yes** | string | Ad set identifier |
| `adId` | **Yes** | string | Ad identifier |
| `destinationUrl` | **Yes** | string | Ad destination URL (used for matching) |
| `name` | No | string | Ad name/title (improves matching) |
| `campaignName` | No | string | Campaign name |
| `adSetName` | No | string | Ad set name |

**Alternative column names accepted:** `ad_name`, `adName`, `destination_url`, `url`

#### 1.2 `products.parquet` or `products.csv`

| Column | Required | Type | Description |
|--------|----------|------|-------------|
| `websiteId` | **Yes** | string | Website identifier (must match ads) |
| `productGroupId` | **Yes** | string | Product group identifier |
| `name` | **Yes** | string | Product name |
| `url` | No | string | Product page URL (improves matching) |
| `productGroupName` | No | string | Product group name |

#### 1.3 `metrics.parquet` or `metrics.csv`

| Column | Required | Type | Description |
|--------|----------|------|-------------|
| `campaignId` | **Yes** | string | Campaign identifier |
| `adSetId` | **Yes** | string | Ad set identifier |
| `adId` | **Yes** | string | Ad identifier |
| `productGroupId` or `productId` | **Yes** | string | Product being attributed |
| `spend` | **Yes** | numeric | Ad spend |
| `impressions` | **Yes** | numeric | Ad impressions |
| `date` | No | date/string | Date (YYYY-MM-DD format) |
| `conversions` | No | numeric | Conversion count |
| `grossProfit` | No | numeric | Revenue/profit |
| `conversionsValue` | No | numeric | Alternative profit column |
| `clicks` | No | numeric | Click count |

### Step 2: Run the Pipeline

#### Basic Usage

```bash
python -m simple.run_pipeline --from-parquet ./data --output-dir ./results
```

#### With Custom Parameters

```bash
python -m simple.run_pipeline \
  --from-parquet ./data \
  --output-dir ./results \
  --format both \
  --fuzzy-threshold 80 \
  --max-products 100 \
  --verbose
```

### Step 3: Understand the Matching Process

The pipeline matches ads to products in 4 stages (in order of priority):

| Stage | Confidence | How It Works |
|-------|------------|--------------|
| **1. exact_url** | 100% | Ad URL exactly matches a product URL |
| **2. exact_product** | 95% | Product slug (e.g., `/products/shoe-pro`) found in ad URL |
| **3. fuzzy** | 85-99% | Ad name fuzzy-matches product name (threshold configurable) |
| **4. token_overlap** | 50% | Keywords overlap between ad and product (fallback) |

If no match is found, the product is marked as **unmatched** (halo/spillover).

### Step 4: Review Output Files

The pipeline creates these files in your output directory:

| File | Description | Use For |
|------|-------------|---------|
| `sku_allocation.parquet/csv` | Product-level metrics with lead/halo flags | **Dashboard input** |
| `targeting.parquet/csv` | Ad → Product mappings with confidence scores | Debugging matches |
| `ad_data.parquet/csv` | Aggregated ad-level totals | Analysis |
| `sku_performance.parquet/csv` | Aggregated SKU-level performance | Analysis |
| `campaign_summary.parquet/csv` | Campaign-level summary | Reporting |

### Step 5: Load into Dashboard

```bash
streamlit run app.py
# Upload results/sku_allocation.csv or .parquet
```

---

## Option B: Metric-Only Pipeline

Use this if you already have a **coalesced product table** (one row per product per ad per day) with pre-mapped ad-to-product relationships.

### Step 1: Prepare Coalesced Product Table

Your input file should have product-level metrics already mapped to ads:

| Column | Required | Type | Description |
|--------|----------|------|-------------|
| `date` or `timestamp` | **Yes** | date/string | Date of metrics |
| `platform` | **Yes** | string | Ad platform |
| `campaignId` | **Yes** | string | Campaign identifier |
| `adSetId` | **Yes** | string | Ad set identifier |
| `adId` | **Yes** | string | Ad identifier |
| `productId` | **Yes** | string | Product identifier |
| `spend` | No | numeric | Ad spend (default: 0) |
| `impressions` | No | numeric | Impressions (default: 0) |
| `clicks` | No | numeric | Clicks (default: 0) |
| `conversions` | No | numeric | Conversions (default: 0) |
| `grossProfit` | No | numeric | Revenue/profit (default: 0) |
| `isLead` | No | 0/1 | Lead product flag (default: 0) |
| `productGroupId` | No | string | Product group ID |
| `productGroupName` | No | string | Product group name |
| `productName` | No | string | Product name |
| `campaignName` | No | string | Campaign name |
| `adSetName` | No | string | Ad set name |
| `adName` | No | string | Ad name |

**Auto-detected alternative column names:**

| Standard | Alternatives Accepted |
|----------|----------------------|
| `date` | `timestamp`, `day` |
| `campaignId` | `campaign_id` |
| `adSetId` | `adsetId`, `ad_set_id`, `adSet_id` |
| `adId` | `ad_id` |
| `productId` | `productId_`, `product_id` |
| `grossProfit` | `gross_profit`, `conversionsValue`, `revenue` |
| `isLead` | `main_product_flag`, `lead_flag` |
| `sku_weight` | `quantity`, `qty`, `units` |

### Step 2: Run the Metric Pipeline

#### Basic Usage

```bash
python getting-metric.py --input ./data/coalesced_df.parquet --output-dir ./results
```

#### With Custom Column Mappings

```bash
python getting-metric.py \
  --input ./data/coalesced_df.parquet \
  --output-dir ./results \
  --profit-col revenue \
  --weight-col quantity \
  --formats csv,parquet
```

### Step 3: Understand Metric Allocation

The pipeline calculates two types of allocations:

| Allocation Type | Description | Column Suffix |
|-----------------|-------------|---------------|
| **Fair** | Metrics split proportionally among ALL products attributed to an ad | `*_fair` |
| **Lead-only** | Metrics allocated ONLY to lead products (halo gets 0) | `*_lead_only` |

**Weight Selection (auto-detected in order):**
1. `sku_weight` (if provided)
2. `quantity`
3. `conversions`
4. `clicks`
5. `impressions`
6. `spend`
7. Default: `1.0` (equal weight)

### Step 4: Load into Dashboard

```bash
streamlit run app.py
# Upload results/sku_allocation.csv
```

---

## Option C: Full Product-Matching Pipeline

Use this for complete end-to-end processing from raw data, including URL processing and product classification.

### Step 1: Prepare Input Parquet Files

Place these 7 files in the repository root:

| File | Description | Key Columns |
|------|-------------|-------------|
| `websites.parquet` | Website configuration | `websiteId`, `name` |
| `df_ads_ori.parquet` | Original ads with extraParams | `websiteId`, `adId`, `destinationUrl`, `extraParams` |
| `df_prod.parquet` | Products | `websiteId`, `productId`, `name`, `url` |
| `df_prod_group.parquet` | Product groups with collections | `websiteId`, `productGroupId`, `collections`, `urls` |
| `df_metrics.parquet` | Product-level metrics | `adId`, `productId`, `spend`, `impressions` |
| `df_metrics_by_country.parquet` | Country-level metrics | `adId`, `country`, `spend`, `impressions` |
| `df_metrics_by_products.parquet` | Supplementary product metrics | `adId`, `productId`, additional metrics |

### Step 2: Run the Full Pipeline

```bash
python product-matching.py
```

**Processing Steps:**
1. Load data (from Parquet or ClickHouse)
2. Process destination URLs (beautify, decode parameters)
3. Process product groups (explode collections, normalize URLs)
4. Enhance ads data (link to processed URLs)
5. Build targeting (ad → product group classification)
6. Classify product tokens (identify main products)
7. Coalesce metrics (combine base and country-level)
8. Compute metrics allocation

### Step 3: Review Output Files

```
results/
├── final_targeting.parquet        # Ad → Product group mappings
├── coalesced_df_with_flags.parquet  # Metrics with isLead flags
└── metric-output/
    ├── ad_data.parquet/csv
    ├── sku_allocation.parquet/csv  ← FOR DASHBOARD
    └── sku_performance.parquet/csv
```

### Step 4: Load into Dashboard

```bash
streamlit run app.py
# Upload results/metric-output/sku_allocation.csv
```

---

## Option D: Manual Data Preparation

Create the dashboard file directly if you have your own attribution system.

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | date/string | Date of the record (YYYY-MM-DD) |
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

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `match_stage` | string | Match confidence: `exact_url`, `exact_product`, `fuzzy`, `token_overlap`, `unmatched` |
| `productId` | string | SKU/product identifier |
| `clicks_fair` | numeric | Clicks allocated to all products |
| `clicks_lead_only` | numeric | Clicks for lead products only |

### Example CSV

```csv
date,productGroupName,productName,platform,campaignId,campaignName,adSetId,adSetName,adId,adName,isLead,gross_profit_fair,spend_fair,impressions_fair,gross_profit_lead_only,spend_lead_only,impressions_lead_only,match_stage
2025-01-15,Shoes,Running Shoe Pro,google,camp_001,Brand Campaign,adset_001,Athletic Audience,ad_001,Running Shoe Ad,1,500.00,100.00,5000,500.00,100.00,5000,exact_url
2025-01-15,Shoes,Casual Sneaker,google,camp_001,Brand Campaign,adset_001,Athletic Audience,ad_001,Running Shoe Ad,0,150.00,30.00,1500,0.00,0.00,0,token_overlap
```

---

## Configurable Parameters Reference

### Simple Pipeline (`simple/run_pipeline.py`)

#### Command Line Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--from-parquet DIR` | - | Load input from Parquet directory |
| `--from-clickhouse` | False | Load input from ClickHouse database |
| `--website-id ID` | `6839260124a2adf314674a5e` | Website ID to filter data |
| `--start-date DATE` | `2025-10-01` | Start date for metrics (YYYY-MM-DD) |
| `--end-date DATE` | `2025-12-31` | End date for metrics (YYYY-MM-DD) |
| `--aws-profile NAME` | `live` | AWS profile for ClickHouse credentials |
| `--fuzzy-threshold INT` | `85` | Fuzzy match score threshold (0-100, higher = stricter) |
| `--max-products INT` | `50` | Maximum products to match per ad |
| `--output-dir PATH` | `./results` | Output directory |
| `--format FORMAT` | `both` | Output format: `csv`, `parquet`, or `both` |
| `--verbose`, `-v` | False | Enable verbose logging |

#### Configuration File (`simple/config.py`)

```python
from simple import PipelineConfig

config = PipelineConfig(
    # Data Source
    website_id="your-website-id",
    start_date="2025-01-01",
    end_date="2025-12-31",

    # ClickHouse Connection (if using --from-clickhouse)
    clickhouse_host="localhost",
    clickhouse_port=8123,
    clickhouse_user="default",
    clickhouse_password="",
    clickhouse_database="profitpeak",
    use_aws_secrets=True,
    aws_profile="live",
    aws_secret_name="SHARED_LAMBDA_CREDENTIALS",
    aws_region="ap-southeast-2",

    # Matching Thresholds
    fuzzy_threshold=85,       # Min fuzzy match score (0-100)
    fuzzy_limit=30,           # Max fuzzy matches per query
    min_token_length=3,       # Min chars for token matching
    max_products_per_ad=50,   # Cap on products matched per ad

    # Output
    output_dir="./results",
    output_formats=["csv", "parquet"],

    # Processing
    max_workers=4,            # Parallel workers
    chunk_size=10000,         # Batch size

    # Column Mappings
    profit_column="conversionsValue",
    weight_column="conversions",
)
```

#### Preset Configurations

```python
from simple.config import default_config, local_config, staging_config

# AWS production config
config = default_config()

# Local ClickHouse (no AWS)
config = local_config()

# Staging environment
config = staging_config()
```

### Metric Pipeline (`getting-metric.py`)

#### Command Line Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input FILE` | **Required** | Input CSV/Parquet file path |
| `--output-dir PATH` | `metric-output` | Output directory |
| `--profit-col COLUMN` | `conversionsValue` | Column for profit/revenue |
| `--weight-col COLUMN` | Auto-detect | Column for allocation weights |
| `--date-col COLUMN` | Auto-detect | Date column override |
| `--platform-col COLUMN` | Auto-detect | Platform column override |
| `--campaign-id-col COLUMN` | Auto-detect | Campaign ID column override |
| `--campaign-name-col COLUMN` | Auto-detect | Campaign name column override |
| `--adset-id-col COLUMN` | Auto-detect | Ad set ID column override |
| `--adset-name-col COLUMN` | Auto-detect | Ad set name column override |
| `--ad-id-col COLUMN` | Auto-detect | Ad ID column override |
| `--ad-name-col COLUMN` | Auto-detect | Ad name column override |
| `--product-id-col COLUMN` | Auto-detect | Product ID column override |
| `--product-group-id-col COLUMN` | Auto-detect | Product group ID column override |
| `--product-name-col COLUMN` | Auto-detect | Product name column override |
| `--product-group-name-col COLUMN` | Auto-detect | Product group name column override |
| `--lead-col COLUMN` | Auto-detect | Lead indicator column override |
| `--sku-weight-col COLUMN` | Auto-detect | SKU weight column override |
| `--formats FORMAT_LIST` | `csv` | Output formats (comma-separated: `csv`, `parquet`) |

### Full Pipeline (`product-matching.py`)

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `URL_MAX_DEPTH` | `5` | Maximum URL redirect depth |
| `URL_MAX_WORKERS` | `8` | Parallel workers for URL processing |
| `URL_CHUNK_SIZE` | `10000` | Batch size for URL processing |
| `CLICKHOUSE_TIMEOUT` | `30` | ClickHouse query timeout (seconds) |
| `RETRY_ATTEMPTS` | `3` | Number of retry attempts for failed operations |
| `CLASSIFICATION_WORKERS` | `4` | Parallel workers for product classification |
| `CLASSIFICATION_CHUNK_SIZE` | `10000` | Batch size for classification |
| `MIN_TOKEN_LEN` | `3` | Minimum token length for matching |
| `FUZZY_THRESHOLD` | `85` | Fuzzy match score threshold |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ENABLE_FILE_LOGGING` | `False` | Enable logging to file |

---

## Using the Dashboard (app.py)

### Starting the Dashboard

```bash
streamlit run app.py
```

### Uploading Data

1. In the **sidebar**, click "Browse files" under "Upload CSV / Parquet"
2. Select your `sku_allocation.csv` or `sku_allocation.parquet` file

### Sidebar Controls

| Control | Description | Default |
|---------|-------------|---------|
| **Min impressions** | Minimum impressions to trust a row | 2000 |
| **Scale if GP/Spend >=** | Efficiency threshold for scaling recommendation | 1.5 |
| **View options** | Campaign-first, ProductGroup-first, or Campaign-only | Campaign-first |
| **Metric scope** | Total (fair), Lead-only, or Halo metrics | Total |
| **Date range** | Filter by date range | All dates |
| **Platform filter** | Filter by ad platform | All platforms |
| **Campaign filter** | Filter by campaign | All campaigns |
| **Product group filter** | Filter by product group | All groups |
| **Match stage filter** | Filter by match confidence | All stages |

### Dashboard Sections

#### 1. Summary Tiles

High-level KPIs:
- **Gross Profit (Total)**: Sum of all gross profit
- **Spend (Total)**: Sum of all ad spend
- **GP / Spend**: Efficiency ratio (higher is better)
- **GP per 1k Impr**: Scalability metric
- **Spillover Share**: % of profit from halo (non-lead) products

#### 2. Actionable Leaderboard

Action recommendations based on performance:

| Priority | Action | Criteria |
|----------|--------|----------|
| 1 | Collect more data | Impressions < min threshold |
| 2 | Scale | High efficiency AND high scale |
| 3 | Keep / Monitor | Good performance |
| 4 | Improve targeting / creative | High efficiency but low scale |
| 5 | Fix economics | Low efficiency but high scale |

#### 3. Quadrant Chart

Visualizes efficiency vs scalability:
- X-axis: GP/Spend (efficiency)
- Y-axis: Impressions or GP per 1k (scalability)
- Bubble size: Total spend

#### 4. Lead vs Halo (Stacked Bar)

Breakdown of metrics between:
- **Lead products**: Directly targeted by the ad
- **Halo products**: Spillover/discovery sales

#### 5. Spillover Insights

Scatter plot and tables showing campaigns with highest/lowest spillover.

#### 6. Trend Analysis

Time series charts for monitoring KPI changes over time.

#### 7. Waste Detection

Identifies where spend is losing money (GP/Spend < 1.0).

---

## Data Schema Reference

### Understanding Lead vs Halo

| Type | Description | isLead | Typical Match Stage |
|------|-------------|--------|---------------------|
| **Lead** | Product directly targeted by ad | 1 | exact_url, exact_product, fuzzy |
| **Halo** | Spillover sale from ad exposure | 0 | token_overlap, unmatched |

### Metric Allocations

| Metric Type | Column Suffix | Description |
|-------------|---------------|-------------|
| **Fair** | `*_fair` | Metrics split proportionally among ALL products attributed to an ad |
| **Lead-only** | `*_lead_only` | Metrics allocated ONLY to lead products (halo gets 0) |

### Match Stages (Confidence Levels)

| Stage | Confidence | Description |
|-------|------------|-------------|
| `exact_url` | 100% | Ad URL exactly matches product URL |
| `exact_product` | 95% | Product slug found in ad URL (e.g., `/products/shoe-pro`) |
| `fuzzy` | 85-99% | Ad name fuzzy-matches product name (configurable threshold) |
| `token_overlap` | 50% | Keywords overlap between ad and product (fallback) |
| `unmatched` | N/A | No match found (classified as halo product) |

---

## Troubleshooting

### "Missing required columns" error

Ensure your file has ALL required columns listed in [Manual Data Preparation](#option-d-manual-data-preparation).

**Common missing columns:**
- `campaignName`, `adSetName`, `adName` (add empty strings if not available)
- `productGroupName` (can use `productName` value if no grouping)

### No products matched in pipeline

1. **Lower fuzzy threshold**: Try `--fuzzy-threshold 70` or `--fuzzy-threshold 75`
2. **Check URLs**: Ensure product URLs are populated and formatted correctly
3. **Verify websiteId**: Must match between ads and products files
4. **Check product names**: Names should be meaningful (not just SKU codes)

### Product-matching workflow cannot find parquet files

1. Confirm all 7 parquet files exist in repository root
2. Check file names match exactly (case-sensitive)
3. Run upstream export/ETL that generates these files

### Dashboard shows "No rows after filters"

1. Check date range filter - expand to include all dates
2. Verify selected platforms exist in your data
3. Remove strict filters temporarily
4. Check if data was uploaded correctly

### Empty charts in dashboard

1. Increase "Top N rows" slider
2. Uncheck "Hide rows with impressions < min"
3. Clear action filters
4. Verify data has non-zero metric values

### Match stage column missing

The dashboard auto-fills `match_stage` as "unmatched" if missing. For better insights, use the simple pipeline which generates this column.

### ClickHouse connection errors

1. Check AWS credentials are configured correctly
2. Verify `--aws-profile` matches your AWS configuration
3. Ensure ClickHouse host is accessible from your network
4. Check timeout settings with `CLICKHOUSE_TIMEOUT` environment variable

### Memory errors with large datasets

1. Use `--chunk-size` to process in smaller batches
2. Filter by date range to reduce data volume
3. Use Parquet format (more memory efficient than CSV)

---

## Quick Start Checklist

- [ ] Prepare input data (ads, products, metrics)
- [ ] Verify all required columns are present
- [ ] Run the pipeline:
  ```bash
  python -m simple.run_pipeline --from-parquet ./data --output-dir ./results
  ```
- [ ] Check output files in `./results`
- [ ] Start dashboard:
  ```bash
  streamlit run app.py
  ```
- [ ] Upload `sku_allocation.csv` or `.parquet`
- [ ] Adjust filters and explore insights

---

## Example Python Usage

```python
from simple import PipelineConfig, AdProductMatcher, compute_all_metrics
import pandas as pd

# 1. Configure the pipeline
config = PipelineConfig(
    website_id="your-website-id",
    fuzzy_threshold=85,
    max_products_per_ad=50,
)

# 2. Load your data
ads_df = pd.read_parquet("data/ads.parquet")
products_df = pd.read_parquet("data/products.parquet")
metrics_df = pd.read_parquet("data/metrics.parquet")

# 3. Build product index and match ads
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

print("Pipeline complete! Upload sku_allocation.csv to the dashboard.")
```
