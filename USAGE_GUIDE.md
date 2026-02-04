# Usage Guide (Simple 3-Step Flow)

This guide keeps the workflow minimal and maps directly to the scripts in this repo.

---

## Step 1: Create the data with `product-matching.py`

**Default ClickHouse + config settings**

Use staging ClickHouse (no AWS credentials needed) and the built-in default config:

```bash
python product-matching.py --from-clickhouse --use-staging
```

This writes results to `./results/`, including:
- `results/coalesced_df_with_flags.parquet` (input for metrics)
- `results/metric-output/sku_allocation.csv` (ready for Streamlit)

**Parameters you can change**

### Data source / ClickHouse
- `--from-parquet` : use local parquet files in repo root instead of ClickHouse.
- `--from-clickhouse` : fetch data from ClickHouse.
- `--use-staging` : use the staging ClickHouse connection (default choice for quick runs).
- `--aws-profile` : AWS profile for production ClickHouse (default: `live`).
- `--website-id` : website ID to process (default: `6839260124a2adf314674a5e`).
- `--start-date` : metrics start date (default: `2025-10-01`).
- `--end-date` : metrics end date (default: `2025-12-31`).
- `--cutoff-date` : cutoff date for metrics coalescing (default: `2025-11-01`).

### Config (environment variables)
These override defaults in `product/config.py`:
- `URL_MAX_DEPTH` (default: `5`)
- `URL_MAX_WORKERS` (default: `8`)
- `URL_CHUNK_SIZE` (default: `10000`)
- `CLICKHOUSE_TIMEOUT` (default: `30`)
- `RETRY_ATTEMPTS` (default: `3`)
- `CLASSIFICATION_WORKERS` (default: `4`)
- `CLASSIFICATION_CHUNK_SIZE` (default: `10000`)
- `COALESCING_TOLERANCE` (default: `1e-9`)
- `MIN_TOKEN_LEN` (default: `3`)
- `FUZZY_THRESHOLD` (default: `85`)
- `LOG_LEVEL` (default: `INFO`)
- `ENABLE_FILE_LOGGING` (default: `False`)

Example with overrides:

```bash
URL_MAX_WORKERS=12 FUZZY_THRESHOLD=90 \
python product-matching.py --from-clickhouse --use-staging --start-date 2025-09-01 --end-date 2025-12-31
```

---

## Step 2: Get metrics with `getting-metric.py`

Use the coalesced output from Step 1:

```bash
python getting-metric.py \
  --input results/coalesced_df_with_flags.parquet \
  --output-dir results/metric-output \
  --formats csv
```

**Parameters you can change**

- `--input` : path to the coalesced table (CSV or Parquet).
- `--output-dir` : where metric files are written (default: `metric-output`).
- `--profit-col` : column to use as profit (default: `conversionsValue`).
- `--weight-col` : column for lead-only weights.
- `--formats` : comma-separated output formats (e.g., `csv,parquet`).

**Column override parameters (only if your input uses different column names)**

- `--date-col`
- `--platform-col`
- `--campaign-id-col`
- `--campaign-name-col`
- `--adset-id-col`
- `--adset-name-col`
- `--ad-id-col`
- `--ad-name-col`
- `--product-id-col`
- `--product-group-id-col`
- `--product-name-col`
- `--product-group-name-col`
- `--lead-col`
- `--sku-weight-col`

---

## Step 3: Use the data in Streamlit

Start the dashboard:

```bash
streamlit run app.py
```

Upload **`results/metric-output/sku_allocation.csv`** (or `.parquet`) when prompted.

**Parameters you can change**

- Streamlit CLI flags, e.g.:
  - `--server.port 8501`
  - `--server.address 0.0.0.0`

Example:

```bash
streamlit run app.py --server.port 8502 --server.address 0.0.0.0
```
