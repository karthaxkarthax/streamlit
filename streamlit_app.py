"""
Innoplus Ticket Analytics — Streamlit App
==========================================
Upload the ticket export (xlsx/csv) in the left panel, filter by Product,
Channel, Country, Language and Customer Category, and review:
  1. Hourly inflow trend (line chart, one line per Product, + auto-generated observations)
  2. Overall SLA / quality summary table (Product x Channel x Category x Country x Language)
  3. Email-channel specific SLA summary table (Product x Category x Country x Language)
  4. KPI observations (quantified, numbers-first bullets)
  5. Recommendations (the 11 standard levers + industry best practices, contextualized
     to the current filters with a quantified KPI impact for each)

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
    "priority": "priority",
    "reply_time_mins": "metric_set.reply_time_in_minutes.business",
    "requester_wait_mins": "metric_set.requester_wait_time_in_minutes.business",
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

# KPI targets used by the observations & recommendations engine.
# CSAT (>=90%) and Reopen (<10%) targets are as specified by the business.
# FRT / 24Hr RT / RWT / Email FRT use a 90% SLA target, the common industry
# convention for first-reply and resolution-time compliance, applied here
# for consistency since no separate target was provided for those metrics.
KPI_TARGETS = {
    "FRT": {"label": "First Reply Time SLA %", "target": 90.0, "higher_is_better": True},
    "24HrRT": {"label": "24-Hr Resolution Time SLA %", "target": 90.0, "higher_is_better": True},
    "CSAT": {"label": "CSAT %", "target": 90.0, "higher_is_better": True},
    "Reopen": {"label": "Reopen %", "target": 10.0, "higher_is_better": False},
    "EmailFRT": {"label": "Email First Reply Time SLA %", "target": 90.0, "higher_is_better": True},
    "RWT": {"label": "Requester Wait Time (24Hr) SLA % [Email]", "target": 90.0, "higher_is_better": True},
}


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


def median_minutes(series: pd.Series) -> float:
    """
    Median elapsed business time in minutes for a raw time metric column.
    Median (rather than mean) is used because these metrics are typically
    heavily right-skewed by a small number of very long-running tickets.
    """
    if series is None:
        return np.nan
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) == 0:
        return np.nan
    return valid.median()


def fmt_minutes(value: float) -> str:
    """Format a minute value as minutes, or as hours+minutes once it's large."""
    if pd.isna(value):
        return "N/A"
    value = float(value)
    if value < 60:
        return f"{value:.0f} min"
    hours = int(value // 60)
    mins = int(round(value % 60))
    return f"{hours}h {mins}m"


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
        row["FRT Median (mins)"] = (
            median_minutes(g[RAW_COLS["reply_time_mins"]]) if RAW_COLS["reply_time_mins"] in g.columns else np.nan
        )
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
        row["Email FRT Median (mins)"] = (
            median_minutes(g[RAW_COLS["reply_time_mins"]]) if RAW_COLS["reply_time_mins"] in g.columns else np.nan
        )
        row["Email FRT Trend"] = (
            trend_arrow(g_sorted, RAW_COLS["email_frt"], pct_compliant)
            if RAW_COLS["email_frt"] in g.columns
            else TREND_FLAT
        )

        row["RWT SLA %"] = rwt_val
        row["RWT Median (mins)"] = (
            median_minutes(g[RAW_COLS["requester_wait_mins"]])
            if RAW_COLS["requester_wait_mins"] in g.columns
            else np.nan
        )
        row["RWT Trend"] = (
            trend_arrow(g_sorted, RAW_COLS["rwt"], pct_compliant) if RAW_COLS["rwt"] in g.columns else TREND_FLAT
        )

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("Ticket Count", ascending=False).head(top_n).reset_index(drop=True)
    return result


def style_table(df: pd.DataFrame, pct_cols=None, min_cols=None) -> "pd.io.formats.style.Styler":
    fmt = {}
    for c in pct_cols or []:
        if c in df.columns:
            fmt[c] = "{:.1f}%"
    for c in min_cols or []:
        if c in df.columns:
            fmt[c] = "{:.0f} min"
    return df.style.format(fmt, na_rep="N/A")


# --------------------------------------------------------------------------
# KPI computation helpers (shared by the Observations & Recommendations engine)
# --------------------------------------------------------------------------
def compute_kpis(sub_df: pd.DataFrame) -> dict:
    """Core KPI set applicable to all channels: FRT, 24Hr RT, CSAT, Reopen %."""
    return {
        "Ticket Count": len(sub_df),
        "FRT": pct_compliant(sub_df[RAW_COLS["frt"]]) if RAW_COLS["frt"] in sub_df.columns else np.nan,
        "FRT_mins": median_minutes(sub_df[RAW_COLS["reply_time_mins"]])
        if RAW_COLS["reply_time_mins"] in sub_df.columns
        else np.nan,
        "24HrRT": pct_compliant(sub_df[RAW_COLS["rt24"]]) if RAW_COLS["rt24"] in sub_df.columns else np.nan,
        "CSAT": pct_csat(sub_df[RAW_COLS["csat"]]) if RAW_COLS["csat"] in sub_df.columns else np.nan,
        "Reopen": pct_reopen(sub_df[RAW_COLS["reopens"]]) if RAW_COLS["reopens"] in sub_df.columns else np.nan,
    }


def compute_email_kpis(df: pd.DataFrame) -> dict:
    """Email-only KPI set: adds Email FRT and Requester Wait Time (RWT), each with a median-minutes figure."""
    if LABELS["channel"] not in df.columns:
        email_df = df.iloc[0:0]
    else:
        email_df = df[df[LABELS["channel"]].astype(str).str.lower() == "email"]
    return {
        "Ticket Count": len(email_df),
        "EmailFRT": pct_compliant(email_df[RAW_COLS["email_frt"]]) if RAW_COLS["email_frt"] in email_df.columns else np.nan,
        "EmailFRT_mins": median_minutes(email_df[RAW_COLS["reply_time_mins"]])
        if RAW_COLS["reply_time_mins"] in email_df.columns
        else np.nan,
        "RWT": pct_compliant(email_df[RAW_COLS["rwt"]]) if RAW_COLS["rwt"] in email_df.columns else np.nan,
        "RWT_mins": median_minutes(email_df[RAW_COLS["requester_wait_mins"]])
        if RAW_COLS["requester_wait_mins"] in email_df.columns
        else np.nan,
    }


def breakdown_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-value KPI breakdown for a single grouping column (e.g. Channel)."""
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    work = df.copy()
    work[group_col] = work[group_col].fillna("Unknown")
    for val, g in work.groupby(group_col, dropna=False):
        k = compute_kpis(g)
        k[group_col] = val
        rows.append(k)
    return pd.DataFrame(rows)


def english_vs_non_english(df: pd.DataFrame):
    if LABELS["language"] not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["_lang_group"] = np.where(work[LABELS["language"]].astype(str) == "English", "English", "Non-English")
    rows = []
    for val, g in work.groupby("_lang_group"):
        k = compute_kpis(g)
        k["Group"] = val
        rows.append(k)
    return pd.DataFrame(rows)


def product_ticket_share(df: pd.DataFrame) -> pd.DataFrame:
    if LABELS["product"] not in df.columns:
        return pd.DataFrame()
    counts = df[LABELS["product"]].fillna("Unknown").value_counts()
    if counts.empty:
        return pd.DataFrame()
    share = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"Product": counts.index, "Ticket Count": counts.values, "Share %": share.values})


def high_priority_kpis(df: pd.DataFrame):
    if RAW_COLS["priority"] not in df.columns:
        return None
    high_df = df[df[RAW_COLS["priority"]].astype(str).str.lower().isin(["high", "urgent"])]
    if high_df.empty:
        return None
    k = compute_kpis(high_df)
    return k


def fmt_pct(v) -> str:
    return f"{v:.1f}%" if pd.notna(v) else "N/A"


def gap_to_target(value: float, kpi_key: str) -> float:
    """Signed distance still needed to reach target; positive means falling short."""
    target = KPI_TARGETS[kpi_key]["target"]
    higher_is_better = KPI_TARGETS[kpi_key]["higher_is_better"]
    if pd.isna(value):
        return np.nan
    return (target - value) if higher_is_better else (value - target)


# --------------------------------------------------------------------------
# Observations engine — every bullet includes at least one number
# --------------------------------------------------------------------------
def generate_kpi_observations(df: pd.DataFrame, overall_table: pd.DataFrame, email_table: pd.DataFrame) -> list:
    obs = []
    overall = compute_kpis(df)
    n = overall["Ticket Count"]

    if pd.notna(overall["FRT"]):
        mins_str = f", median reply time **{fmt_minutes(overall['FRT_mins'])}**" if pd.notna(overall["FRT_mins"]) else ""
        obs.append(
            f"**First Reply Time (FRT) SLA** is at **{overall['FRT']:.1f}%** across {n:,} tickets"
            f"{mins_str} "
            f"({gap_to_target(overall['FRT'],'FRT'):+.1f} points vs. the 90% target)."
        )
    if pd.notna(overall["24HrRT"]):
        obs.append(
            f"**24-Hr Resolution Time SLA** is at **{overall['24HrRT']:.1f}%** "
            f"({gap_to_target(overall['24HrRT'],'24HrRT'):+.1f} points vs. the 90% target)."
        )
    if pd.notna(overall["CSAT"]):
        obs.append(
            f"**CSAT** is at **{overall['CSAT']:.1f}%** against the 90% target "
            f"({gap_to_target(overall['CSAT'],'CSAT'):+.1f} points)."
        )
    if pd.notna(overall["Reopen"]):
        obs.append(
            f"**Reopen rate** is at **{overall['Reopen']:.1f}%** against the <10% target "
            f"({gap_to_target(overall['Reopen'],'Reopen'):+.1f} points over target)."
        )

    # Channel spread
    ch_kpis = breakdown_by(df, LABELS["channel"]) if LABELS["channel"] in df.columns else pd.DataFrame()
    if not ch_kpis.empty and len(ch_kpis) > 1 and ch_kpis["FRT"].notna().sum() > 1:
        best = ch_kpis.loc[ch_kpis["FRT"].idxmax()]
        worst = ch_kpis.loc[ch_kpis["FRT"].idxmin()]
        obs.append(
            f"FRT SLA varies by channel: **{best[LABELS['channel']]}** leads at **{best['FRT']:.1f}%** "
            f"vs. **{worst[LABELS['channel']]}** at **{worst['FRT']:.1f}%**, a gap of "
            f"**{best['FRT']-worst['FRT']:.1f} points**."
        )

    # Non-English gap
    lang_kpis = english_vs_non_english(df)
    if not lang_kpis.empty and set(lang_kpis["Group"]) == {"English", "Non-English"}:
        eng = lang_kpis[lang_kpis["Group"] == "English"].iloc[0]
        non_eng = lang_kpis[lang_kpis["Group"] == "Non-English"].iloc[0]
        if pd.notna(eng["CSAT"]) and pd.notna(non_eng["CSAT"]):
            obs.append(
                f"Non-English tickets ({non_eng['Ticket Count']:,} tickets) show CSAT of "
                f"**{non_eng['CSAT']:.1f}%** vs. **{eng['CSAT']:.1f}%** for English tickets "
                f"({eng['Ticket Count']:,} tickets), a **{eng['CSAT']-non_eng['CSAT']:+.1f} point** gap."
            )

    # Trend direction count from the overall table
    if not overall_table.empty and "FRT Trend" in overall_table.columns:
        down_count = (overall_table["FRT Trend"] == TREND_DOWN).sum()
        up_count = (overall_table["FRT Trend"] == TREND_UP).sum()
        total_rows = len(overall_table)
        obs.append(
            f"Among the top {total_rows} Channel/Product/Category/Country/Language combinations, "
            f"**{down_count}** show a declining FRT trend and **{up_count}** show an improving FRT trend "
            f"over the selected period."
        )

    # Email-specific
    if not email_table.empty:
        em = compute_email_kpis(df)
        if pd.notna(em["EmailFRT"]):
            mins_str = (
                f", median reply time **{fmt_minutes(em['EmailFRT_mins'])}**" if pd.notna(em["EmailFRT_mins"]) else ""
            )
            obs.append(
                f"Email channel First Reply Time SLA is **{em['EmailFRT']:.1f}%** across "
                f"**{em['Ticket Count']:,}** email tickets{mins_str}."
            )
        if pd.notna(em["RWT"]):
            mins_str = f", median wait time **{fmt_minutes(em['RWT_mins'])}**" if pd.notna(em["RWT_mins"]) else ""
            obs.append(
                f"Email channel Requester Wait Time (24Hr) SLA is **{em['RWT']:.1f}%**{mins_str} "
                f"({gap_to_target(em['RWT'],'RWT'):+.1f} points vs. the 90% target)."
            )

    return obs


# --------------------------------------------------------------------------
# Recommendations engine — links each generic recommendation to the current
# filtered data, and quantifies at least one KPI that would improve.
# --------------------------------------------------------------------------
def generate_recommendations(df: pd.DataFrame, overall_table: pd.DataFrame, email_table: pd.DataFrame) -> list:
    recs = []
    overall = compute_kpis(df)
    ch_kpis = breakdown_by(df, LABELS["channel"]) if LABELS["channel"] in df.columns else pd.DataFrame()
    lang_kpis = english_vs_non_english(df)
    prod_share = product_ticket_share(df)
    email_kpis = compute_email_kpis(df)
    hp_kpis = high_priority_kpis(df)

    # 1. Smart channel routing
    if not ch_kpis.empty and len(ch_kpis) > 1:
        scored = ch_kpis.copy()
        scored["_score"] = scored[["FRT", "24HrRT", "CSAT"]].mean(axis=1, skipna=True)
        scored = scored.dropna(subset=["_score"])
        if len(scored) > 1:
            best = scored.loc[scored["_score"].idxmax()]
            worst = scored.loc[scored["_score"].idxmin()]
            gap = best["_score"] - worst["_score"]
            if gap >= 2:
                recs.append(
                    (
                        "1. Smart channel routing",
                        f"**{best[LABELS['channel']]}** tickets average **{best['_score']:.1f}%** across FRT/24Hr RT/CSAT, "
                        f"vs. **{worst['_score']:.1f}%** for **{worst[LABELS['channel']]}** — a **{gap:.1f} point** gap. "
                        f"Routing more eligible {worst[LABELS['channel']]} volume through {best[LABELS['channel']]}-style "
                        f"handling (or the channel itself, where feasible) could lift {worst[LABELS['channel']]}'s FRT SLA "
                        f"from **{worst['FRT']:.1f}%** toward **{best['FRT']:.1f}%**.",
                    )
                )
            else:
                recs.append(
                    (
                        "1. Smart channel routing",
                        f"Channel performance is fairly even currently (composite scores within **{gap:.1f} points** "
                        f"of each other across {len(scored)} channels), so routing gains would be marginal today — "
                        f"worth re-checking as volumes shift.",
                    )
                )

    # 2. Non-English AI real-time translation
    if not lang_kpis.empty and set(lang_kpis["Group"]) >= {"English", "Non-English"}:
        eng = lang_kpis[lang_kpis["Group"] == "English"].iloc[0]
        non_eng = lang_kpis[lang_kpis["Group"] == "Non-English"].iloc[0]
        if pd.notna(eng["FRT"]) and pd.notna(non_eng["FRT"]) and non_eng["Ticket Count"] > 0:
            frt_gap = eng["FRT"] - non_eng["FRT"]
            mins_str = (
                f" (median reply time **{fmt_minutes(non_eng['FRT_mins'])}** vs. **{fmt_minutes(eng['FRT_mins'])}** "
                f"for English)"
                if pd.notna(non_eng["FRT_mins"]) and pd.notna(eng["FRT_mins"])
                else ""
            )
            recs.append(
                (
                    "2. AI real-time translation for non-English cases",
                    f"Non-English tickets (**{non_eng['Ticket Count']:,}** of {int(overall['Ticket Count']):,}, "
                    f"{non_eng['Ticket Count']/overall['Ticket Count']*100:.1f}%) run FRT SLA of "
                    f"**{non_eng['FRT']:.1f}%** vs. **{eng['FRT']:.1f}%** for English "
                    f"({frt_gap:+.1f} point gap){mins_str}. Real-time translation could help close this gap toward "
                    f"the 90% FRT SLA target.",
                )
            )

    # 3. AI emailBOT for email composition
    if pd.notna(email_kpis["EmailFRT"]) and email_kpis["Ticket Count"] > 0:
        mins_str = (
            f", median reply time **{fmt_minutes(email_kpis['EmailFRT_mins'])}**"
            if pd.notna(email_kpis["EmailFRT_mins"])
            else ""
        )
        recs.append(
            (
                "3. AI emailBOT for reply drafting",
                f"Email channel First Reply Time SLA is **{email_kpis['EmailFRT']:.1f}%** across "
                f"**{email_kpis['Ticket Count']:,}** email tickets{mins_str} "
                f"({gap_to_target(email_kpis['EmailFRT'],'EmailFRT'):+.1f} points vs. the 90% target). "
                f"An AI-drafted reply for agent review/edit/send could reduce drafting time and help close this gap.",
            )
        )

    # 4. Lead notification before FRT breach
    if pd.notna(overall["FRT"]):
        gap = gap_to_target(overall["FRT"], "FRT")
        mins_str = f", median reply time **{fmt_minutes(overall['FRT_mins'])}**" if pd.notna(overall["FRT_mins"]) else ""
        recs.append(
            (
                "4. Proactive lead alert near FRT threshold",
                f"Current FRT SLA is **{overall['FRT']:.1f}%**{mins_str} ({gap:+.1f} points vs. the 90% target). "
                f"Alerting leads when a case is approaching its FRT threshold — before it breaches — is a "
                f"common early-warning pattern that could reduce the number of missed replies contributing to this gap.",
            )
        )

    # 5. Knowledge base with SME/trainer support
    if pd.notna(overall["Reopen"]):
        recs.append(
            (
                "5. SME/trainer-built knowledge base",
                f"Reopen rate is currently **{overall['Reopen']:.1f}%** (target <10%). A structured knowledge base, "
                f"built and maintained with SME/trainer input, would give agents standardized resolution guidance — "
                f"typically improving first-time-fix rates and helping bring reopen rate down over time.",
            )
        )

    # 6. Similar/closed-ticket retrieval
    if pd.notna(overall["24HrRT"]):
        recs.append(
            (
                "6. Real-time retrieval of similar closed tickets",
                f"24-Hr Resolution SLA is **{overall['24HrRT']:.1f}%** "
                f"({gap_to_target(overall['24HrRT'],'24HrRT'):+.1f} points vs. target). Surfacing similar, "
                f"already-resolved tickets to agents at case-open time would let them reuse proven resolutions, "
                f"which typically shortens resolution time and supports this SLA.",
            )
        )

    # 7. Skill-based agent routing (language/channel/product)
    if not ch_kpis.empty and len(ch_kpis) > 1 and ch_kpis["CSAT"].notna().sum() > 1:
        csat_spread = ch_kpis["CSAT"].max() - ch_kpis["CSAT"].min()
        recs.append(
            (
                "7. Skill-based agent routing",
                f"CSAT varies by **{csat_spread:.1f} points** across channels in the current selection. "
                f"Routing tickets to agents matched on language, channel and product proficiency is a standard "
                f"lever to narrow this spread and lift CSAT toward the 90% target.",
            )
        )

    # 8. Shift-left for high-frequency issues per product
    if not prod_share.empty:
        top_prod = prod_share.iloc[0]
        if top_prod["Share %"] >= 30:
            recs.append(
                (
                    "8. Shift-left for high-frequency issues",
                    f"**{top_prod['Product']}** accounts for **{top_prod['Share %']:.1f}%** "
                    f"({top_prod['Ticket Count']:,} tickets) of current volume. Identifying its highest-frequency "
                    f"issue types and shifting them left (self-service, guided flows, or automation) could reduce "
                    f"incoming volume and free up capacity to raise FRT/24Hr RT SLA elsewhere.",
                )
            )

    # 9. Best-practice sharing / mentoring from high performers
    if not ch_kpis.empty and len(ch_kpis) > 1 and ch_kpis["FRT"].notna().sum() > 1:
        best = ch_kpis.loc[ch_kpis["FRT"].idxmax()]
        recs.append(
            (
                "9. Best-practice sharing & mentoring",
                f"**{best[LABELS['channel']]}** currently leads on FRT SLA at **{best['FRT']:.1f}%**. "
                f"Documenting the practices behind that performance and pairing lower performers with mentors "
                f"is a standard way to lift the group average toward that benchmark.",
            )
        )

    # 10. Process workflow study for automation
    if pd.notna(overall["24HrRT"]):
        recs.append(
            (
                "10. Workflow study for end-to-end automation",
                f"With 24-Hr Resolution SLA at **{overall['24HrRT']:.1f}%**, mapping the end-to-end ticket workflow "
                f"to find manual, automatable steps is a standard throughput lever — even a modest reduction in "
                f"handling steps typically compounds into measurable SLA gains at this ticket volume "
                f"({int(overall['Ticket Count']):,} tickets).",
            )
        )

    # 11. Temporary fixes for high-priority tickets
    if hp_kpis is not None and pd.notna(hp_kpis["24HrRT"]):
        recs.append(
            (
                "11. Temporary fixes for high-priority tickets",
                f"High/Urgent priority tickets ({hp_kpis['Ticket Count']:,} tickets) show 24-Hr Resolution SLA of "
                f"**{hp_kpis['24HrRT']:.1f}%**. Offering a temporary fix where applicable — de-escalating priority "
                f"while the permanent fix is worked — could help this cohort's SLA move toward the 90% target faster.",
            )
        )
    elif RAW_COLS["priority"] not in df.columns:
        recs.append(
            (
                "11. Temporary fixes for high-priority tickets",
                f"The uploaded file doesn't include a `priority` column, so this can't be quantified against the "
                f"current {int(overall['Ticket Count']):,}-ticket base — once available, tracking High/Urgent SLA "
                f"specifically (against the 90% target) would show where temporary fixes help most.",
            )
        )

    # Additional industry-standard best practices
    if pd.notna(overall["CSAT"]):
        recs.append(
            (
                "12. Tiered (L1/L2/L3) support model",
                f"With CSAT at **{overall['CSAT']:.1f}%**, escalation tiering — routine issues resolved at L1, "
                f"complex cases reserved for specialists — is an industry-standard structure that typically "
                f"improves both FRT (faster routine handling) and CSAT (better-matched specialist attention).",
            )
        )
    if pd.notna(overall["Reopen"]):
        recs.append(
            (
                "13. Root-cause analysis on reopened tickets",
                f"Reopen rate is **{overall['Reopen']:.1f}%**. A lightweight RCA step on every reopened ticket "
                f"(standard in mature support orgs) helps identify systemic fix gaps and is typically one of the "
                f"fastest levers to bring reopen rate down toward the <10% target.",
            )
        )
    recs.append(
        (
            "14. QA scorecards & calibration sessions",
            f"Regular quality scorecards and calibration sessions across agents/teams (current base: "
            f"{int(overall['Ticket Count']):,} tickets in scope) are an industry-standard mechanism to keep "
            f"CSAT and SLA performance consistent as volume grows.",
        )
    )

    return recs


# --------------------------------------------------------------------------
# Sidebar — upload + filters
# --------------------------------------------------------------------------
st.sidebar.title("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload ticket export (.xlsx or .csv)", type=["xlsx", "xls", "csv"])

st.title("Cyncly Support Analytics")

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
    min_cols = ["FRT Median (mins)"]
    st.dataframe(style_table(overall_table, pct_cols, min_cols), use_container_width=True, height=500)

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
    pct_cols = ["Email FRT SLA %", "RWT SLA %"]
    min_cols = ["Email FRT Median (mins)", "RWT Median (mins)"]
    st.dataframe(style_table(email_table, pct_cols, min_cols), use_container_width=True, height=500)

# --------------------------------------------------------------------------
# 4. KPI observations (every bullet includes at least one number)
# --------------------------------------------------------------------------
st.header("🔍 KPI Observations")

kpi_observations = generate_kpi_observations(df, overall_table, email_table)
if kpi_observations:
    st.markdown("\n".join(f"- {o}" for o in kpi_observations))
else:
    st.info("Not enough data in the current selection to generate KPI observations.")

# --------------------------------------------------------------------------
# 5. Recommendations, linked to the current filtered condition
# --------------------------------------------------------------------------
st.header("💡 Recommendations")

active_filters = []
if sel_product:
    active_filters.append(f"Product: {', '.join(sel_product)}")
if sel_channel:
    active_filters.append(f"Channel: {', '.join(sel_channel)}")
if sel_country:
    active_filters.append(f"Country: {', '.join(sel_country)}")
if sel_language:
    active_filters.append(f"Language: {', '.join(sel_language)}")
if sel_category:
    active_filters.append(f"Customer Category: {', '.join(sel_category)}")

if active_filters:
    st.caption("Based on the current filters — " + " | ".join(active_filters) + f" — {len(df):,} tickets in scope.")
else:
    st.caption(f"Based on the full dataset (no filters applied) — {len(df):,} tickets in scope.")

recommendations = generate_recommendations(df, overall_table, email_table)
if recommendations:
    for title, body in recommendations:
        with st.expander(title):
            st.markdown(body)
else:
    st.info("Not enough data in the current selection to generate recommendations.")

