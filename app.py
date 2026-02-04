# app.py
# Streamlit BI (merchant-friendly) for campaign × productGroupName performance with lead vs halo (spillover)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ----------------------------
# Constants
# ----------------------------
DEFAULT_PLATFORMS = ["google", "meta", "tiktok", "pinterest"]

ACTION_CHOICES = [
    "1. Collect more data",
    "2. Scale",
    "3. Keep / Monitor",
    "4. Improve targeting / creative",
    "5. Fix economics (price/AOV) or tighten audience",
]


# ----------------------------
# Helpers
# ----------------------------
def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace({0: np.nan})
    return numer / denom


def coerce_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def require_columns(df: pd.DataFrame, required) -> list:
    return [c for c in required if c not in df.columns]


def normalize_is_lead(series: pd.Series) -> pd.Series:
    # Accepts 0/1, True/False, "0"/"1", "true"/"false"
    s = series.copy()
    if s.dtype == bool:
        return s.astype(int)
    s = s.astype(str).str.strip().str.lower()
    return s.map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}).fillna(0).astype(int)


def aggregate_kpis(df: pd.DataFrame, dims: list) -> pd.DataFrame:
    """Aggregation-safe KPIs computed from SUMs (prevents wrong totals)."""
    agg = df.groupby(dims, dropna=False).agg(
        gp_total=("gross_profit_fair", "sum"),
        spend_total=("spend_fair", "sum"),
        impr_total=("impressions_fair", "sum"),
        gp_lead=("gross_profit_lead_only", "sum"),
        spend_lead=("spend_lead_only", "sum"),
        impr_lead=("impressions_lead_only", "sum"),
    ).reset_index()

    agg["gp_halo"] = agg["gp_total"] - agg["gp_lead"]
    agg["spend_halo"] = agg["spend_total"] - agg["spend_lead"]
    agg["impr_halo"] = agg["impr_total"] - agg["impr_lead"]

    agg["gp_eff_total"] = safe_div(agg["gp_total"], agg["spend_total"])
    agg["gp_eff_lead"] = safe_div(agg["gp_lead"], agg["spend_lead"])
    agg["gp_eff_halo"] = safe_div(agg["gp_halo"], agg["spend_halo"])

    agg["gp_per_1k_total"] = 1000 * safe_div(agg["gp_total"], agg["impr_total"])
    agg["gp_per_1k_lead"] = 1000 * safe_div(agg["gp_lead"], agg["impr_lead"])
    agg["gp_per_1k_halo"] = 1000 * safe_div(agg["gp_halo"], agg["impr_halo"])

    agg["cpm_total"] = 1000 * safe_div(agg["spend_total"], agg["impr_total"])
    agg["spillover_share"] = safe_div(agg["gp_halo"], agg["gp_total"])

    agg["spillover_ratio_guarded"] = np.where(
        agg["gp_lead"] < 50,
        np.nan,
        safe_div(agg["gp_halo"], agg["gp_lead"])
    )
    return agg


def action_label_numbered(
    gp_eff: pd.Series,
    gp_per_1k: pd.Series,
    impr: pd.Series,
    min_impr: int,
    scale_eff: float,
    scale_gp1k: float,
    ok_eff: float,
    ok_gp1k: float,
) -> pd.Series:
    conds = [
        impr < min_impr,
        (gp_eff >= scale_eff) & (gp_per_1k >= scale_gp1k),
        (gp_eff >= ok_eff) & (gp_per_1k >= ok_gp1k),
        (gp_eff >= ok_eff) & (gp_per_1k < ok_gp1k),
        (gp_eff < ok_eff) & (gp_per_1k >= ok_gp1k),
    ]
    choices = [
        "1. Collect more data",
        "2. Scale",
        "3. Keep / Monitor",
        "4. Improve targeting / creative",
        "5. Fix economics (price/AOV) or tighten audience",
    ]
    return pd.Series(np.select(conds, choices, default="4. Improve targeting / creative"), index=gp_eff.index)


def sorted_platform_defaults_first(platform_list):
    rest = [p for p in platform_list if p not in DEFAULT_PLATFORMS]
    return DEFAULT_PLATFORMS + sorted(rest)


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Campaign × Product Group BI (Lead vs Halo)", layout="wide")
st.title("Campaign × Product Group BI (Lead vs Halo / Spillover)")
st.caption("Explore efficiency (GP/Spend), scale (GP per 1k impressions), and spillover (halo).")

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV / Parquet", type=["csv", "parquet"])

    st.divider()
    st.header("Scoring thresholds")
    min_impr = st.number_input("Min impressions to trust a row", min_value=0, value=2000, step=500)
    scale_eff = st.number_input("Scale if GP/Spend ≥", min_value=0.0, value=1.30, step=0.05, format="%.2f")
    scale_gp1k = st.number_input("Scale if GP per 1k Impr ≥", min_value=0.0, value=2.00, step=0.25, format="%.2f")
    ok_eff = st.number_input("OK if GP/Spend ≥", min_value=0.0, value=1.00, step=0.05, format="%.2f")
    ok_gp1k = st.number_input("OK if GP per 1k Impr ≥", min_value=0.0, value=1.00, step=0.25, format="%.2f")

    st.caption("Scoring rules (based on your thresholds):")
    st.markdown(
        f"""
- **1. Collect more data**  
  If **Impressions < {int(min_impr):,}**

- **2. Scale**  
  If **Impressions ≥ {int(min_impr):,}** & **GP/Spend ≥ {scale_eff:.2f}** & **GP per 1k Impr ≥ {scale_gp1k:.2f}**

- **3. Keep / Monitor**  
  If **Impressions ≥ {int(min_impr):,}** & **GP/Spend ≥ {ok_eff:.2f}** & **GP per 1k Impr ≥ {ok_gp1k:.2f}**  
  *(but not meeting “Scale” thresholds above)*

- **4. Improve targeting / creative**  
  If **Impressions ≥ {int(min_impr):,}** & **GP/Spend ≥ {ok_eff:.2f}** & **GP per 1k Impr < {ok_gp1k:.2f}**

- **5. Fix economics (price/AOV) or tighten audience**  
  If **Impressions ≥ {int(min_impr):,}** & **GP/Spend < {ok_eff:.2f}** & **GP per 1k Impr ≥ {ok_gp1k:.2f}**
"""
    )

    st.divider()
    st.header("View options")
    view_mode = st.radio("Primary view", ["Campaign-first", "ProductGroup-first", "Campaign-only"], index=2)
    metric_mode = st.radio("Metric scope (for plots)", ["Total (fair)", "Lead-only", "Halo"], index=0)
    top_n = st.slider("Top N rows", 10, 1000000, 50, 10)
    include_product_name = st.checkbox("Optional: include productName drill-down", value=False)

    st.divider()
    st.header("Page filters")


@st.cache_data(show_spinner=False)
def load_df(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    if uploaded.name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    return pd.read_csv(uploaded)


df = load_df(up)

required_cols = [
    "date",
    "productGroupName",
    "productName",
    "platform",
    "campaignId", "campaignName",
    "adSetId", "adSetName",
    "adId", "adName",
    "isLead",
    "gross_profit_fair", "spend_fair", "impressions_fair",
    "gross_profit_lead_only", "spend_lead_only", "impressions_lead_only",
]

# Optional columns (will be created if missing)
optional_cols = ["match_stage"]

if df.empty:
    st.info("Upload a CSV/Parquet to start. Column names must match your output schema.")
    st.stop()

missing = require_columns(df, required_cols)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = coerce_date(df, "date")
df["isLead_norm"] = normalize_is_lead(df["isLead"])

# Handle optional match_stage column (for confidence tracking of product group match)
if "match_stage" not in df.columns:
    df["match_stage"] = "unmatched"
else:
    df["match_stage"] = df["match_stage"].fillna("unmatched").astype(str)

# ----------------------------
# Sidebar filters (after load)
# ----------------------------
with st.sidebar:
    # Date filter
    min_date = pd.to_datetime(df["date"]).min()
    max_date = pd.to_datetime(df["date"]).max()
    date_range = None
    if not (pd.isna(min_date) or pd.isna(max_date)):
        date_range = st.date_input("Date range", value=(min_date.date(), max_date.date()))
    else:
        st.warning("Some 'date' values could not be parsed. Date filter disabled.")

    platforms_all = sorted([p for p in df["platform"].dropna().unique().tolist()])
    platforms_ordered = sorted_platform_defaults_first(platforms_all)
    default_platform_sel = [p for p in DEFAULT_PLATFORMS if p in platforms_all]
    platform_sel = st.multiselect("Platform", platforms_ordered, default=default_platform_sel)

    campaign_names = sorted([c for c in df["campaignName"].dropna().unique().tolist()])
    camp_sel = st.multiselect("Campaign", campaign_names, default=[])

    pg_names = sorted([g for g in df["productGroupName"].dropna().unique().tolist()])
    pg_sel = st.multiselect("Product group", pg_names, default=[])

    islead_sel = st.multiselect("isLead (0/1)", [0, 1], default=[0, 1])

    # Match stage filter (confidence of product group match)
    match_stages_all = sorted([s for s in df["match_stage"].dropna().unique().tolist()])
    match_stage_sel = st.multiselect(
        "Match stage (confidence)",
        match_stages_all,
        default=match_stages_all,
        help="Filter by product-ad match confidence: exact_url (highest) > exact_product > fuzzy > token_overlap > unmatched (halo)"
    )

    remove_null_campaign = st.checkbox("Remove NULL/blank campaignName", value=True)
    
# Apply filters
df_f = df.copy()
if date_range:
    start_d, end_d = date_range
    df_f = df_f[
        (pd.to_datetime(df_f["date"]) >= pd.to_datetime(start_d)) &
        (pd.to_datetime(df_f["date"]) <= pd.to_datetime(end_d))
    ]
df_f = df_f[df_f["platform"].isin(platform_sel)]
if camp_sel:
    df_f = df_f[df_f["campaignName"].isin(camp_sel)]
if pg_sel:
    df_f = df_f[df_f["productGroupName"].isin(pg_sel)]
if islead_sel:
    df_f = df_f[df_f["isLead_norm"].isin(islead_sel)]
if match_stage_sel:
    df_f = df_f[df_f["match_stage"].isin(match_stage_sel)]

if remove_null_campaign:
    df_f = df_f[df_f["campaignName"].notna()]
    df_f = df_f[df_f["campaignName"].astype(str).str.strip() != ""]


if df_f.empty:
    st.warning("No rows after filters.")
    st.stop()

# Choose metric columns based on metric_mode
if metric_mode.startswith("Total"):
    eff_col, gp1k_col, gp_col, spend_col, impr_col = "gp_eff_total", "gp_per_1k_total", "gp_total", "spend_total", "impr_total"
elif metric_mode.startswith("Lead"):
    eff_col, gp1k_col, gp_col, spend_col, impr_col = "gp_eff_lead", "gp_per_1k_lead", "gp_lead", "spend_lead", "impr_lead"
else:
    eff_col, gp1k_col, gp_col, spend_col, impr_col = "gp_eff_halo", "gp_per_1k_halo", "gp_halo", "spend_halo", "impr_halo"

# ----------------------------
# Summary tiles (2 lines)
# ----------------------------
st.subheader("Summary")

tot_gp = float(df_f["gross_profit_fair"].sum())
tot_spend = float(df_f["spend_fair"].sum())
tot_impr = float(df_f["impressions_fair"].sum())
tot_eff = (tot_gp / tot_spend) if tot_spend else np.nan
tot_gp1k = (1000 * tot_gp / tot_impr) if tot_impr else np.nan
tot_spill_share = ((df_f["gross_profit_fair"].sum() - df_f["gross_profit_lead_only"].sum()) / tot_gp) if tot_gp else np.nan



# Line 1: core financial KPIs
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
r1c1.metric("Gross Profit (Total)", f"{tot_gp:,.0f}")
r1c2.metric("Spend (Total)", f"{tot_spend:,.0f}")
r1c3.metric("GP / Spend", f"{tot_eff:,.2f}" if pd.notna(tot_eff) else "—")
r1c4.metric("GP per 1k Impr", f"{tot_gp1k:,.2f}" if pd.notna(tot_gp1k) else "—")
r1c5.metric("Spillover Share", f"{tot_spill_share*100:,.1f}%" if pd.notna(tot_spill_share) else "—")

# Line 2: impressions-related KPIs
median_grain = st.selectbox(
    "Impressions stats grain (median/avg)",
    ["campaign", "adset", "ad"],
    index=0,
    help="Median/Avg impressions across entities in this grain (within current filters)."
)

grain_map = {
    "campaign": ["campaignId", "campaignName"],
    "adset": ["campaignId", "campaignName", "adSetId", "adSetName"],
    "ad": ["campaignId", "campaignName", "adSetId", "adSetName", "adId", "adName"],
}
median_dims = grain_map[median_grain]
impr_stats_df = aggregate_kpis(df_f, dims=median_dims)

median_impr = float(impr_stats_df["impr_total"].median()) if not impr_stats_df.empty else np.nan
avg_impr = float(impr_stats_df["impr_total"].mean()) if not impr_stats_df.empty else np.nan



r2c1, r2c2, r2c3 = st.columns(3)
r2c1.metric("Impressions (Total)", f"{tot_impr:,.0f}")
r2c2.metric(f"Median Impr ({median_grain})", f"{median_impr:,.0f}" if pd.notna(median_impr) else "—")
r2c3.metric(f"Avg Impr ({median_grain})", f"{avg_impr:,.0f}" if pd.notna(avg_impr) else "—")

st.divider()


# ----------------------------
# 1) Actionable leaderboard (Table)
# ----------------------------
st.subheader("1) Actionable leaderboard")

# Action filter per section
action_sel_1 = st.multiselect("Filter actions (Leaderboard)", ACTION_CHOICES, default=ACTION_CHOICES, key="action_sel_1")

if view_mode == "Campaign-only":
    default_dims = ["platform", "campaignName", "adSetName", "adName"]
elif view_mode == "ProductGroup-first":
    default_dims = ["platform", "campaignName", "adSetName", "adName", "productGroupName"]
else:
    default_dims = ["platform", "productGroupName", "campaignName", "adSetName", "adName"]
if include_product_name:
    default_dims = default_dims + ["productName"]

dims = st.multiselect(
    "Group by dimensions",
    options=[
        "platform",
        "campaignId", "campaignName",
        "adSetId", "adSetName",
        "adId", "adName",
        "productGroupName",
        "productName" if include_product_name else None,
        "match_stage",
    ],
    default=[d for d in default_dims if d is not None],
    key="leader_dims"
)
dims = [d for d in dims if d]
if not dims:
    st.warning("Select at least one group-by dimension.")
    st.stop()

agg = aggregate_kpis(df_f, dims=dims)
agg["action_next"] = action_label_numbered(
    gp_eff=agg[eff_col],
    gp_per_1k=agg[gp1k_col],
    impr=agg[impr_col],
    min_impr=int(min_impr),
    scale_eff=float(scale_eff),
    scale_gp1k=float(scale_gp1k),
    ok_eff=float(ok_eff),
    ok_gp1k=float(ok_gp1k),
)
if action_sel_1:
    agg = agg[agg["action_next"].isin(action_sel_1)]

only_sufficient = st.checkbox(f"Hide rows with {impr_col} < min impressions", value=True, key="leader_hide_low")
if only_sufficient:
    agg = agg[agg[impr_col] >= int(min_impr)]

# Metric columns selector (default = current)
default_metric_cols = [gp_col, spend_col, impr_col, eff_col, gp1k_col, "spillover_share", "action_next"]
metric_cols = st.multiselect(
    "Metric columns to display",
    options=[
        "gp_total","spend_total","impr_total","gp_eff_total","gp_per_1k_total","cpm_total",
        "gp_lead","spend_lead","impr_lead","gp_eff_lead","gp_per_1k_lead",
        "gp_halo","spend_halo","impr_halo","gp_eff_halo","gp_per_1k_halo",
        "spillover_share","spillover_ratio_guarded","action_next"
    ],
    default=default_metric_cols,
    key="leader_metric_cols"
)

agg = agg.sort_values(by=[eff_col, gp1k_col, gp_col], ascending=False).head(int(top_n))
show_cols = dims + metric_cols
show_cols = [c for c in show_cols if c in agg.columns]
st.dataframe(agg[show_cols], use_container_width=True, hide_index=True)

st.divider()

# ----------------------------
# 2) Quadrant: Efficient vs Scalable (scatter)
# ----------------------------
st.subheader("2) Quadrant: Efficient vs Scalable (scatter)")

action_sel_2 = st.multiselect(
    "Filter actions (Scatter)",
    ACTION_CHOICES,
    default=ACTION_CHOICES,
    key="action_sel_2"
)

scatter_dims = ["platform"]
scatter_dim = "campaignName" if view_mode == "Campaign-first" or view_mode else "productGroupName"
scatter_dims.append(scatter_dim)
if include_product_name:
    scatter_dims.append("productName")

scatter_df = aggregate_kpis(df_f, dims=scatter_dims)

# Always compute action based on current metric scope (x/y), but always filter by TOTAL impressions for stability
scatter_df["action_next"] = action_label_numbered(
    gp_eff=scatter_df[eff_col],
    gp_per_1k=scatter_df[gp1k_col],
    impr=scatter_df[impr_col],
    min_impr=int(min_impr),
    scale_eff=float(scale_eff),
    scale_gp1k=float(scale_gp1k),
    ok_eff=float(ok_eff),
    ok_gp1k=float(ok_gp1k),
)

if action_sel_2:
    scatter_df = scatter_df[scatter_df["action_next"].isin(action_sel_2)]

# Stability guard: always require total impressions (prevents insane y + weird bubble)
scatter_df = scatter_df[scatter_df["impr_total"] >= int(min_impr)]

if scatter_df.empty:
    st.info("No points after filters for scatter.")
else:
    # Bubble sizing
    scatter_df = scatter_df.copy()
    scatter_df["bubble_size"] = scatter_df["spend_total"]

    # Build scatter
    fig_scatter = px.scatter(
        scatter_df,
        x=eff_col,
        y=impr_col,
        size="bubble_size",
        size_max=45,
        color="platform",
        hover_name=scatter_dim,
        hover_data={
            "platform": True,
            "spend_total": ":,.0f",     # required: totalSpend on hover
            "gp_total": ":,.0f",
            "impr_total": ":,.0f",
            "spillover_share": ".1%",
            "action_next": True,
            # hide helper + avoid duplicate noisy fields
            "bubble_size": False,
        },
        title=f"{metric_mode}: {impr_col} vs {eff_col} (bubble size = totalSpend)",
    )

    # Make hover clean + consistent (prevents %{customdata[x]} weirdness)
    fig_scatter.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Platform=%{customdata[0]}<br>"
            f"{eff_col}=%{{x:.2f}}<br>"
            f"{gp1k_col}=%{{y:,.2f}}<br>"
            "totalSpend=$%{customdata[1]:,.0f}<br>"
            "totalGP=$%{customdata[2]:,.0f}<br>"
            "totalImpr=%{customdata[3]:,.0f}<br>"
            "spilloverShare=%{customdata[4]}<br>"
            "action=%{customdata[5]}<extra></extra>"
        )
    )

    # Ensure customdata order matches hover_data order above
    # Plotly express sets customdata in the order of hover_data keys that are not False
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ----------------------------
# 3) Lead vs Halo  (WORKING + clean hover + correct sorting)
# ----------------------------
st.subheader("3) Lead vs Halo")

action_sel_3 = st.multiselect(
    "Filter actions (Lead vs Halo)",
    ACTION_CHOICES,
    default=ACTION_CHOICES,
    key="action_sel_3"
)

# Campaign-level aggregation
camp = aggregate_kpis(df_f, dims=["platform", "campaignName"])
camp["action_next"] = action_label_numbered(
    gp_eff=camp["gp_eff_total"],
    gp_per_1k=camp["gp_per_1k_total"],
    impr=camp["impr_total"],
    min_impr=int(min_impr),
    scale_eff=float(scale_eff),
    scale_gp1k=float(scale_gp1k),
    ok_eff=float(ok_eff),
    ok_gp1k=float(ok_gp1k),
)

if action_sel_3:
    camp = camp[camp["action_next"].isin(action_sel_3)]

# Platform focus
platforms_avail = sorted([p for p in camp["platform"].dropna().unique().tolist()])
platform_focus = st.selectbox(
    "Platform",
    ["All"] + platforms_avail,
    index=0,
    key="platform_focus_lead_halo"
)

camp_v = camp.copy()
if platform_focus != "All":
    camp_v = camp_v[camp_v["platform"] == platform_focus]

# Controls
cA, cB, cC = st.columns([1, 1, 1])
with cA:
    n_show = st.slider("Show top N campaigns", 5, 50, 20, 1, key="lead_halo_n")
with cB:
    sort_by = st.selectbox(
        "Sort by",
        ["gp_total", "spend_total", "gp_eff_total", "gp_per_1k_total", "impr_total", "spillover_share"],
        index=0,
        key="lead_halo_sort"
    )
with cC:
    metric_split = st.selectbox("Split metric", ["Gross Profit", "Spend", "Impression"], index=0, key="lead_halo_metric")

hide_low_data = st.checkbox("Hide low-impression campaigns", value=True, key="lead_halo_hide_low")
if hide_low_data:
    camp_v = camp_v[camp_v["impr_total"] >= int(min_impr)]

if camp_v.empty:
    st.info("No campaigns after filters for this chart.")
else:
    # Choose what to stack
    if metric_split == "Gross Profit":
        v_lead, v_halo = "gp_lead", "gp_halo"
        title = "Lead vs Halo — Gross Profit (stacked)"
        x_tickprefix, x_tickformat = "$", ",.0f"
    elif metric_split == "Spend":
        v_lead, v_halo = "spend_lead", "spend_halo"
        title = "Lead vs Halo — Spend (stacked)"
        x_tickprefix, x_tickformat = "$", ",.0f"
    else:
        v_lead, v_halo = "impr_lead", "impr_halo"
        title = "Lead vs Halo — Impression (stacked)"
        x_tickprefix, x_tickformat = "", ",.0f"

    # Sort and take top N (this determines bar order)
    camp_v = camp_v.sort_values(sort_by, ascending=False).head(int(n_show)).copy()

    # Avoid duplicate campaign names when Platform = All
    if platform_focus == "All":
        camp_v["campaign_label"] = camp_v["platform"].astype(str) + " | " + camp_v["campaignName"].astype(str)
    else:
        camp_v["campaign_label"] = camp_v["campaignName"].astype(str)

    campaign_order = camp_v["campaign_label"].tolist()[::-1]  # top-to-bottom

    # IMPORTANT: do NOT include v_lead/v_halo in id_vars, otherwise melt drops them => empty chart
    long_df = camp_v.melt(
        id_vars=[
            "platform", "campaignName", "campaign_label", "action_next",
            "gp_total", "spend_total", "impr_total",
            "gp_eff_total", "gp_per_1k_total", "spillover_share",
        ],
        value_vars=[v_lead, v_halo],
        var_name="segment_raw",
        value_name="value"
    )

    segment_map = {v_lead: "Lead", v_halo: "Halo"}
    long_df["segment"] = long_df["segment_raw"].map(segment_map)

    fig = px.bar(
        long_df,
        y="campaign_label",
        x="value",
        color="segment",
        orientation="h",
        category_orders={"campaign_label": campaign_order},
        title=title,
    )

    # Clean, stable hover (no weird auto fields)
    # customdata index:
    # 0 platform, 1 campaignName, 2 action, 3 gp_total, 4 spend_total, 5 impr_total, 6 spillover_share, 7 segment
    fig.update_traces(
        customdata=np.stack([
            long_df["platform"].astype(str),
            long_df["campaignName"].astype(str),
            long_df["action_next"].astype(str),
            long_df["gp_total"].astype(float),
            long_df["spend_total"].astype(float),
            long_df["impr_total"].astype(float),
            long_df["spillover_share"].astype(float),
            long_df["segment"].astype(str),
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Platform=%{customdata[0]}<br>"
            "Action=%{customdata[2]}<br>"
            "Segment=%{customdata[7]}<br>"
            f"{'Gross Profit' if metric_split=='Gross Profit' else 'Spend'}=$%{{x:,.0f}}<br>"
            "Total GP=$%{customdata[3]:,.0f}<br>"
            "Total Spend=$%{customdata[4]:,.0f}<br>"
            "Total Impr=%{customdata[5]:,.0f}<br>"
            "Spillover Share=%{customdata[6]:.1%}"
            "<extra></extra>"
        )
    )

    fig.update_layout(height=max(450, 30 * int(n_show)), legend_title_text="")
    fig.update_xaxes(tickprefix=x_tickprefix, tickformat=x_tickformat)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show campaign table (readable)"):
        st.dataframe(
            camp_v[[
                "platform", "campaignName", "action_next",
                "gp_total", "spend_total", "impr_total",
                "gp_eff_total", "gp_per_1k_total", "spillover_share",
                "gp_lead", "gp_halo", "spend_lead", "spend_halo"
            ]],
            use_container_width=True,
            hide_index=True
        )


st.divider()
# ----------------------------
# 4) Spillover insights (scatter + top/bottom list)
# ----------------------------
st.subheader("4) Spillover insights (scatter + top/bottom list)")

action_sel_4 = st.multiselect("Filter actions (Spillover)", ACTION_CHOICES, default=ACTION_CHOICES, key="action_sel_4")

spill = aggregate_kpis(df_f, dims=["platform", "campaignName"])
spill["action_next"] = action_label_numbered(
    gp_eff=spill["gp_eff_total"],
    gp_per_1k=spill["gp_per_1k_total"],
    impr=spill["impr_total"],
    min_impr=int(min_impr),
    scale_eff=float(scale_eff),
    scale_gp1k=float(scale_gp1k),
    ok_eff=float(ok_eff),
    ok_gp1k=float(ok_gp1k),
)
if action_sel_4:
    spill = spill[spill["action_next"].isin(action_sel_4)]

platforms_sp = sorted([p for p in spill["platform"].dropna().unique().tolist()])
platform_focus2 = st.selectbox("Platform (spillover)", ["All"] + platforms_sp, index=0, key="platform_focus2")
if platform_focus2 != "All":
    spill = spill[spill["platform"] == platform_focus2]

spill = spill[spill["impr_total"] >= int(min_impr)]

if spill.empty:
    st.info("No campaigns after filters for spillover chart.")
else:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        y_choice = st.selectbox("Y-axis", ["gp_eff_total", "gp_per_1k_total", "gp_total"], index=0, key="spill_y")
    with c2:
        size_choice = st.selectbox("Bubble size", ["spend_total", "gp_total", "impr_total"], index=0, key="spill_size")
    with c3:
        show_n = st.slider("Top/Bottom list size", 5, 50, 15, 5, key="spill_n")

    fig2 = px.scatter(
        spill,
        x="spillover_share",
        y=y_choice,
        size=size_choice,
        color="platform" if platform_focus2 == "All" else None,
        hover_data=["campaignName", "action_next", "gp_total", "spend_total", "impr_total", "gp_lead", "gp_halo"],
        title="Spillover Share vs Performance (bubble size = scale)",
    )
    fig2.update_layout(height=520)
    fig2.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    left, right = st.columns(2)
    top_df = spill.sort_values("spillover_share", ascending=False).head(int(show_n))
    bot_df = spill.sort_values("spillover_share", ascending=True).head(int(show_n))

    with left:
        st.markdown("**Highest Spillover Share (halo-heavy / discovery-like)**")
        st.dataframe(
            top_df[["platform","campaignName","spillover_share","gp_total","spend_total","gp_eff_total","gp_per_1k_total","action_next"]],
            use_container_width=True, hide_index=True
        )
    with right:
        st.markdown("**Lowest Spillover Share (lead-product driven)**")
        st.dataframe(
            bot_df[["platform","campaignName","spillover_share","gp_total","spend_total","gp_eff_total","gp_per_1k_total","action_next"]],
            use_container_width=True, hide_index=True
        )

st.divider()

# ----------------------------
# 5) Trend (time series)
# ----------------------------
st.subheader("5) Trend (time series)")

action_sel_5 = st.multiselect("Filter actions (Trend)", ACTION_CHOICES, default=ACTION_CHOICES, key="action_sel_5")

breakdown = st.selectbox("Break down by", ["None", "platform", "campaignName", "productGroupName"], index=1, key="trend_breakdown")
ts_dims = ["date"] + ([] if breakdown == "None" else [breakdown])

ts = aggregate_kpis(df_f, dims=ts_dims).sort_values("date")
ts["action_next"] = action_label_numbered(
    gp_eff=ts["gp_eff_total"],
    gp_per_1k=ts["gp_per_1k_total"],
    impr=ts["impr_total"],
    min_impr=int(min_impr),
    scale_eff=float(scale_eff),
    scale_gp1k=float(scale_gp1k),
    ok_eff=float(ok_eff),
    ok_gp1k=float(ok_gp1k),
)

filter_ts_by_action = st.checkbox("Filter time series by selected actions", value=False, key="trend_filter_action")
if filter_ts_by_action and action_sel_5:
    ts = ts[ts["action_next"].isin(action_sel_5)]

metric_for_ts = st.selectbox(
    "Metric",
    options=[
        ("GP / Spend (Total)", "gp_eff_total"),
        ("GP per 1k Impr (Total)", "gp_per_1k_total"),
        ("Gross Profit (Total)", "gp_total"),
        ("Spillover Share", "spillover_share"),
    ],
    index=0,
    format_func=lambda x: x[0],
    key="trend_metric"
)[1]

if breakdown == "None":
    fig_ts = px.line(ts, x="date", y=metric_for_ts, title=f"Trend: {metric_for_ts}")
else:
    fig_ts = px.line(ts, x="date", y=metric_for_ts, color=breakdown, title=f"Trend: {metric_for_ts} by {breakdown}")
st.plotly_chart(fig_ts, use_container_width=True)

st.divider()

# ----------------------------
# 6) Waste (where spend is losing money)
# ----------------------------
st.subheader("6) Waste (where spend is losing money)")

# Action filter only applicable here (per your note)
action_sel_6 = st.multiselect("Filter actions (Waste)", ACTION_CHOICES, default=ACTION_CHOICES, key="action_sel_6")

default_waste_dims = ["platform", "campaignName", "adSetName", "adName"]
if include_product_name:
    default_waste_dims.append("productName")

waste_dims = st.multiselect(
    "Dimensions (Waste table)",
    options=[
        "platform",
        "campaignId", "campaignName",
        "adSetId", "adSetName",
        "adId", "adName",
        "productGroupName",
        "productName" if include_product_name else None,
        "match_stage",
    ],
    default=[d for d in default_waste_dims if d is not None and 'product' not in d],
    key="waste_dims"
)
waste_dims = [d for d in waste_dims if d]
if not waste_dims:
    st.warning("Select at least one dimension for Waste.")
    st.stop()

w = aggregate_kpis(df_f, dims=waste_dims)
w["action_next"] = action_label_numbered(
    gp_eff=w["gp_eff_total"],
    gp_per_1k=w["gp_per_1k_total"],
    impr=w["impr_total"],
    min_impr=int(min_impr),
    scale_eff=float(scale_eff),
    scale_gp1k=float(scale_gp1k),
    ok_eff=float(ok_eff),
    ok_gp1k=float(ok_gp1k),
)
if action_sel_6:
    w = w[w["action_next"].isin(action_sel_6)]

w["wasted_spend"] = np.where(w["gp_eff_total"] < 1.0, w["spend_total"], 0.0)
w = w[w["wasted_spend"] > 0].sort_values("wasted_spend", ascending=False).head(int(top_n))

st.dataframe(
    w[waste_dims + ["wasted_spend", "gp_total", "impr_total", "gp_eff_total", "gp_per_1k_total", "spillover_share", "action_next"]],
    use_container_width=True,
    hide_index=True
)

st.caption("Default platform selection starts with: google, meta, tiktok, pinterest (others can be added in the Platform filter).")
