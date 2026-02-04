# Metrics calculations

This document describes how the metric outputs are derived from the coalesced product table.

## Inputs

The metrics pipeline starts with a coalesced product table and applies a column mapping step.
By default, the mapping looks for common column names (e.g., `timestamp` → `date`,
`main_product_flag` → `isLead`, `conversionsValue` → `grossProfit`), and it can be overridden
when calling `compute_metrics()` or `getting-metric.py`.

After normalization, the pipeline expects these **required** columns:

- `date` (or `timestamp`, which is converted to `date`)
- `platform`
- `campaignId`
- `adSetId`
- `adId`
- `productId` (or `productId_`)

These columns are **optional but supported** (defaults filled if missing):

- `campaignName`, `adSetName`, `adName`
- `productGroupId`, `productGroupName`, `productName`
- `spend`, `impressions`, `clicks`, `interactions`, `conversions`
- A profit column (default `grossProfit`, but `conversionsValue` is commonly used)
- A lead indicator column (`isLead` or `main_product_flag`)
- An allocation weight column (see weight selection below)

If an optional column is missing, it is filled with zeroes or empty strings during normalization.

## Normalization

1. **Date**
   - `date` is converted to a day-level date (`YYYY-MM-DD`) using `timestamp` when needed.
2. **Identifiers**
   - All IDs (`campaignId`, `adSetId`, `adId`, `productId`, `productGroupId`) are cast to strings.
3. **Metrics**
   - Numeric metrics (`spend`, `impressions`, `clicks`, `interactions`, `conversions`, `grossProfit`)
     are coerced to numbers and missing values become `0`.
4. **Lead flag**
   - `isLead` is derived from `lead_col` (if provided), otherwise `isLead`, otherwise
     `main_product_flag` (via the column map). Valid truthy values: `1`, `true`, `yes`.
5. **Profit column**
   - `grossProfit` is populated from the configured profit column (default `grossProfit`;
     `getting-metric.py` uses `conversionsValue` by default).
6. **Weight**
   - `sku_weight` is selected from the first available column in this order:
     1. `sku_weight`
     2. `quantity`
     3. `conversions`
     4. `clicks`
     5. `impressions`
     6. `spend`
     7. default `1.0`

## Output 1: `ad_data`

Per-ad totals, derived by summing over each `date × platform × campaignId × adSetId × adId` group.

- `ad_spend_total = Σ spend`
- `ad_impressions_total = Σ impressions`
- `ad_clicks_total = Σ clicks`
- `ad_gross_profit_total = Σ grossProfit`

## Output 2: `sku_allocation`

Allocation of each ad’s totals to individual SKUs.

1. **Aggregate at SKU level**
   - Group by `date × productId × productGroupId × platform × campaignId × adSetId × adId × isLead`
   - Sum: `spend`, `impressions`, `clicks`, `grossProfit`, `sku_weight`

2. **Compute weights within each ad**
   - `total_weight_all = Σ sku_weight`
   - `lead_weight = sku_weight if isLead = 1 else 0`
   - `total_weight_lead = Σ lead_weight`

3. **Shares**
   - `share = sku_weight / total_weight_all` (or `0` if `total_weight_all = 0`)
   - `share_lead_only = lead_weight / total_weight_lead` (or `0` if `total_weight_lead = 0`)

4. **Fair allocation (all SKUs)**
   - `spend_fair = share × ad_spend_total`
   - `impressions_fair = share × ad_impressions_total`
   - `clicks_fair = share × ad_clicks_total`
   - `gross_profit_fair = share × ad_gross_profit_total`

5. **Lead-only allocation**
   - `spend_lead_only = share_lead_only × ad_spend_total`
   - `impressions_lead_only = share_lead_only × ad_impressions_total`
   - `clicks_lead_only = share_lead_only × ad_clicks_total`
   - `gross_profit_lead_only = share_lead_only × ad_gross_profit_total`

## Output 3: `sku_performance`

Aggregated SKU-level totals from `sku_allocation`, grouped by `date × productId`.

- `sku_spend_fair = Σ spend_fair`
- `sku_impressions_fair = Σ impressions_fair`
- `sku_clicks_fair = Σ clicks_fair`
- `sku_gross_profit_fair = Σ gross_profit_fair`
- `sku_spend_lead_only = Σ spend_lead_only`
- `sku_impressions_lead_only = Σ impressions_lead_only`
- `sku_clicks_lead_only = Σ clicks_lead_only`
- `sku_gross_profit_lead_only = Σ gross_profit_lead_only`
