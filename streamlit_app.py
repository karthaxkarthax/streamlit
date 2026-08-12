"""
Cyncly Support Ticket Analytics — Streamlit App
==========================================
Upload the ticket export (xlsx/csv) in the left panel, filter by Product,
Channel, Country, Language and Customer Category, and review:
  1. Hourly inflow trend (line chart, one line per Product, + auto-generated observations)
  2. Overall SLA / quality summary table (Product x Channel x Category x Country x Language)
  3. Email-channel specific SLA summary table (Product x Category x Country x Language)

Run with:  streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Innoplus Ticket Analytics", layout="wide")

# --------------------------------------------------------------------------
# Column name map (source column -> internal / display label)
# --------------------------------------------------------------------------
RAW_COLS = {
    "product": "group.name",
    "channel": "via.channel",
    "country": "new country",
    "language": "new language",
    "category": "new customer category",
    "hour": "new Hour",
    "month": "new Month",
    "created_at": "created_at",
    "frt": "New FRT (1-Compliant)",
    "email_frt": "New Email FRT",
    "rt24": "New 24Hrs RT (1-Com)",
    "rwt": "New RWT (1-Comp)",
    "csat": "satisfaction_rating.score",
    "reopens": "metric_set.reopens",
}

LABELS = {
    "product": "Product",
    "channel": "Channel",
    "country": "Country",
    "language": "Language",
    "category": "Customer Category",
}

TREND_UP, TREND_DOWN, TREND_FLAT = "\u2191", "\u2193", "\u2192"
TREND_THRESHOLD = 1.0  # percentage points; smaller moves are shown as flat


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # keep only columns we recognise from the raw map that actually exist,
    # but don't drop anything else — just rename the filter columns for display
    rename_map = {
        RAW_COLS["product"]: LABELS["product"],
        RAW_COLS["channel"]: LABELS["channel"],
        RAW_COLS["country"]: LABELS["country"],
        RAW_COLS["language"]: LABELS["language"],
        RAW_COLS["category"]: LABELS["category"],
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if RAW_COLS["created_at"] in df.columns:
        df["_created_dt"] = pd.to_datetime(df[RAW_COLS["created_at"]], errors="coerce")
    else:
        df["_created_dt"] = pd.NaT

    return df


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def pct_compliant(series: pd.Series) -> float:
    """% of 1s among non-blank values of a 0/1/blank column."""
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) == 0:
        return np.nan
    return (valid == 1).sum() / len(valid) * 100


def pct_csat(series: pd.Series) -> float:
    """CSAT% = good / (good + bad)."""
    good = (series == "good").sum()
    bad = (series == "bad").sum()
    denom = good + bad
    if denom == 0:
        return np.nan
    return good / denom * 100


def pct_reopen(series: pd.Series) -> float:
    """Reopen rate = % of records with reopens > 0."""
    valid = pd.to_numeric(series, errors="coerce")
    total = valid.notna().sum()
    if total == 0:
        return np.nan
    return (valid.fillna(0) > 0).sum() / total * 100


def trend_arrow(group_sorted: pd.DataFrame, col: str, metric_func, higher_is_better=True) -> str:
    """
    Compare the metric value in the first half vs second half of the
    date-sorted group (split at the midpoint) to infer a trend over the
    whole period.
    """
    valid_dates = group_sorted["_created_dt"].notna().sum()
    if len(group_sorted) < 6 or valid_dates < 6:
        return TREND_FLAT
    mid = len(group_sorted) // 2
    first_half = group_sorted.iloc[:mid]
    second_half = group_sorted.iloc[mid:]
    v1 = metric_func(first_half[col])
    v2 = metric_func(second_half[col])
    if pd.isna(v1) or pd.isna(v2):
        return TREND_FLAT
    diff = v2 - v1
    if abs(diff) < TREND_THRESHOLD:
        return TREND_FLAT
    improving = diff > 0 if higher_is_better else diff < 0
    return TREND_UP if improving else TREND_DOWN


def status_badge(value: float, target_ok) -> str:
    if pd.isna(value):
        return "N/A"
    return "✅ Compliant" if target_ok(value) else "🔴 Breach"


# --------------------------------------------------------------------------
# Summary table builders
# --------------------------------------------------------------------------
def build_overall_summary(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    group_cols = [LABELS["product"], LABELS["channel"], LABELS["category"], LABELS["country"], LABELS["language"]]
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    work = df.copy()
    for c in group_cols:
        work[c] = work[c].fillna("Unknown")

    rows = []
    for keys, g in work.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        g_sorted = g.sort_values("_created_dt")
        row = dict(zip(group_cols, keys))
        row["Ticket Count"] = len(g)

        frt_val = pct_compliant(g[RAW_COLS["frt"]]) if RAW_COLS["frt"] in g.columns else np.nan
        rt24_val = pct_compliant(g[RAW_COLS["rt24"]]) if RAW_COLS["rt24"] in g.columns else np.nan
        csat_val = pct_csat(g[RAW_COLS["csat"]]) if RAW_COLS["csat"] in g.columns else np.nan
        reopen_val = pct_reopen(g[RAW_COLS["reopens"]]) if RAW_COLS["reopens"] in g.columns else np.nan

        row["FRT SLA %"] = frt_val
        row["FRT Trend"] = (
            trend_arrow(g_sorted, RAW_COLS["frt"], pct_compliant) if RAW_COLS["frt"] in g.columns else TREND_FLAT
        )

        row["24Hr RT SLA %"] = rt24_val
        row["24Hr RT Trend"] = (
            trend_arrow(g_sorted, RAW_COLS["rt24"], pct_compliant) if RAW_COLS["rt24"] in g.columns else TREND_FLAT
        )

        row["CSAT %"] = csat_val
        row["CSAT Trend"] = (
            trend_arrow(g_sorted, RAW_COLS["csat"], pct_csat) if RAW_COLS["csat"] in g.columns else TREND_FLAT
        )
        row["CSAT Status (>=90%)"] = status_badge(csat_val, lambda v: v >= 90)

        row["Reopen Rate %"] = reopen_val
        row["Reopen Trend"] = (
            trend_arrow(g_sorted, RAW_COLS["reopens"], pct_reopen, higher_is_better=False)
            if RAW_COLS["reopens"] in g.columns
            else TREND_FLAT
        )
        row["Reopen Status (<10%)"] = status_badge(reopen_val, lambda v: v < 10)

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("Ticket Count", ascending=False).head(top_n).reset_index(drop=True)
    return result


def build_email_summary(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if LABELS["channel"] not in df.columns:
        return pd.DataFrame()

    email_df = df[df[LABELS["channel"]].astype(str).str.lower() == "email"].copy()
    if email_df.empty:
        return pd.DataFrame()

    group_cols = [LABELS["product"], LABELS["category"], LABELS["country"], LABELS["language"]]
    group_cols = [c for c in group_cols if c in email_df.columns]
    if not group_cols:
        return pd.DataFrame()

    for c in group_cols:
        email_df[c] = email_df[c].fillna("Unknown")

    rows = []
    for keys, g in email_df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        g_sorted = g.sort_values("_created_dt")
        row = dict(zip(group_cols, keys))
        row["Ticket Count"] = len(g)

        efrt_val = pct_compliant(g[RAW_COLS["email_frt"]]) if RAW_COLS["email_frt"] in g.columns else np.nan
        rwt_val = pct_compliant(g[RAW_COLS["rwt"]]) if RAW_COLS["rwt"] in g.columns else np.nan

        row["Email FRT SLA %"] = efrt_val
        row["Email FRT Trend"] = (
            trend_arrow(g_sorted, RAW_COLS["email_frt"], pct_compliant)
            if RAW_COLS["email_frt"] in g.columns
            else TREND_FLAT
        )

        row["24Hr RT (RWT) SLA %"] = rwt_val
        row["24Hr RT (RWT) Trend"] = (
            trend_arrow(g_sorted, RAW_COLS["rwt"], pct_compliant) if RAW_COLS["rwt"] in g.columns else TREND_FLAT
        )

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("Ticket Count", ascending=False).head(top_n).reset_index(drop=True)
    return result


def style_pct_cols(df: pd.DataFrame, pct_cols) -> "pd.io.formats.style.Styler":
    fmt = {c: "{:.1f}%" for c in pct_cols if c in df.columns}
    return df.style.format(fmt, na_rep="N/A")


# --------------------------------------------------------------------------
# Sidebar — upload + filters
# --------------------------------------------------------------------------
st.sidebar.title("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload ticket export (.xlsx or .csv)", type=["xlsx", "xls", "csv"])

st.title("Cyncly Support Ticket Analytics")

if uploaded_file is None:
    st.info("Upload a data file from the left panel to get started.")
    st.stop()

with st.spinner("Loading data..."):
    df_raw = load_data(uploaded_file)

st.sidebar.markdown("---")
st.sidebar.title("🔎 Filters")


def multiselect_filter(label: str, col: str, df: pd.DataFrame):
    if col not in df.columns:
        return None
    options = sorted(df[col].dropna().astype(str).unique().tolist())
    return st.sidebar.multiselect(label, options, default=[])


sel_product = multiselect_filter(LABELS["product"], LABELS["product"], df_raw)
sel_channel = multiselect_filter(LABELS["channel"], LABELS["channel"], df_raw)
sel_country = multiselect_filter(LABELS["country"], LABELS["country"], df_raw)
sel_language = multiselect_filter(LABELS["language"], LABELS["language"], df_raw)
sel_category = multiselect_filter(LABELS["category"], LABELS["category"], df_raw)

df = df_raw.copy()
if sel_product:
    df = df[df[LABELS["product"]].astype(str).isin(sel_product)]
if sel_channel:
    df = df[df[LABELS["channel"]].astype(str).isin(sel_channel)]
if sel_country:
    df = df[df[LABELS["country"]].astype(str).isin(sel_country)]
if sel_language:
    df = df[df[LABELS["language"]].astype(str).isin(sel_language)]
if sel_category:
    df = df[df[LABELS["category"]].astype(str).isin(sel_category)]

st.caption(f"Showing **{len(df):,}** of {len(df_raw):,} tickets after filters.")

if df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# --------------------------------------------------------------------------
# 1. Hourly inflow line chart + observations
# --------------------------------------------------------------------------
st.header("⏱️ Hourly Ticket Inflow")

if RAW_COLS["hour"] in df.columns:
    has_product = LABELS["product"] in df.columns

    if has_product:
        hourly_by_product = (
            df.assign(**{LABELS["product"]: df[LABELS["product"]].fillna("Unknown")})
            .groupby([RAW_COLS["hour"], LABELS["product"]])
            .size()
            .reset_index(name="Ticket Count")
            .rename(columns={RAW_COLS["hour"]: "Hour"})
        )
        # ensure every product has all 24 hours represented (fills gaps with 0)
        all_hours = pd.DataFrame({"Hour": range(24)})
        hourly_by_product = (
            hourly_by_product.groupby(LABELS["product"])
            .apply(lambda g: all_hours.merge(g, on="Hour", how="left").assign(**{LABELS["product"]: g.name}))
            .reset_index(drop=True)
        )
        hourly_by_product["Ticket Count"] = hourly_by_product["Ticket Count"].fillna(0)

        fig = px.line(
            hourly_by_product,
            x="Hour",
            y="Ticket Count",
            color=LABELS["product"],
            markers=True,
            title="Ticket Inflow by Hour of Day, by Product",
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

        # totals across all products, used for the observations below
        hourly_df = hourly_by_product.groupby("Hour")["Ticket Count"].sum().reindex(range(24), fill_value=0)
        hourly_df = hourly_df.reset_index()
        hourly_df.columns = ["Hour", "Ticket Count"]
    else:
        hourly_counts = df.groupby(RAW_COLS["hour"]).size().reindex(range(24), fill_value=0)
        hourly_df = hourly_counts.reset_index()
        hourly_df.columns = ["Hour", "Ticket Count"]

        fig = px.line(
            hourly_df,
            x="Hour",
            y="Ticket Count",
            markers=True,
            title="Ticket Inflow by Hour of Day",
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    total = int(hourly_df["Ticket Count"].sum())
    peak_hour = int(hourly_df.loc[hourly_df["Ticket Count"].idxmax(), "Hour"])
    peak_val = int(hourly_df["Ticket Count"].max())
    low_hour = int(hourly_df.loc[hourly_df["Ticket Count"].idxmin(), "Hour"])
    low_val = int(hourly_df["Ticket Count"].min())

    top3 = hourly_df.sort_values("Ticket Count", ascending=False).head(3)
    top3_hours = ", ".join(f"{int(h)}:00" for h in top3["Hour"])
    top3_share = (top3["Ticket Count"].sum() / total * 100) if total else 0

    business_hours = hourly_df[(hourly_df["Hour"] >= 8) & (hourly_df["Hour"] <= 18)]
    business_share = (business_hours["Ticket Count"].sum() / total * 100) if total else 0

    st.markdown("**Observations:**")
    st.markdown(
        f"""
- Peak inflow occurs at **{peak_hour}:00** with **{peak_val:,}** tickets; the quietest hour is **{low_hour}:00** with **{low_val:,}** tickets.
- The top 3 busiest hours (**{top3_hours}**) together account for **{top3_share:.1f}%** of all ticket inflow.
- **{business_share:.1f}%** of tickets arrive during core business hours (08:00–18:00), suggesting staffing should be concentrated in this window.
"""
    )
else:
    st.warning(f"Column '{RAW_COLS['hour']}' not found in the uploaded data.")

# --------------------------------------------------------------------------
# 2. Overall SLA / quality summary table
# --------------------------------------------------------------------------
st.header("📊 Overall SLA & Quality Summary")
st.caption(
    "Grouped by Product, Channel, Customer Category, Country and Language — top 50 combinations by ticket count. "
    "Trend arrows compare the first half vs. second half of the selected date range."
)

overall_table = build_overall_summary(df)
if overall_table.empty:
    st.warning("Not enough data / required columns to build this table.")
else:
    pct_cols = ["FRT SLA %", "24Hr RT SLA %", "CSAT %", "Reopen Rate %"]
    st.dataframe(style_pct_cols(overall_table, pct_cols), use_container_width=True, height=500)

# --------------------------------------------------------------------------
# 3. Email-channel specific SLA summary table
# --------------------------------------------------------------------------
st.header("📧 Email Channel SLA Summary")
st.caption(
    "Restricted to via.channel = 'email'. Grouped by Product, Customer Category, Country and Language — "
    "top 50 combinations by ticket count."
)

email_table = build_email_summary(df)
if email_table.empty:
    st.info("No 'email' channel records found in the current filter selection.")
else:
    pct_cols = ["Email FRT SLA %", "24Hr RT (RWT) SLA %"]
    st.dataframe(style_pct_cols(email_table, pct_cols), use_container_width=True, height=500)
