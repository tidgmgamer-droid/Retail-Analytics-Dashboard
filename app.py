"""
Enterprise Retail Intelligence Dashboard
==========================================
Navigation  : st.tabs — zero-latency switching
Filters     : Sidebar multiselect + age slider + date range + reset button
Caching     : @st.cache_data (data) · @st.cache_resource (Prophet model)
Theme       : Glassmorphism neon KPI cards · plotly_dark · transparent bg
Prophet fix : cutoff passed as pd.Timestamp directly to Scatter trace;
              add_vline() is NEVER used (avoids Plotly datetime-sum bug)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Enterprise Retail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────── */
.stApp { background-color: #0a0e1a; color: #e6edf3; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2733;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e6edf3 !important; }

/* ── Tabs ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid #1e2733;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: #0d1117;
    border: 1px solid #1e2733;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    color: #6e7681;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 9px 22px;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #c9d1d9 !important; }
.stTabs [aria-selected="true"] {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border-top: 2px solid #00b4ff !important;
    border-bottom-color: #161b22 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent;
    padding-top: 1rem;
}

/* ── Glassmorphism KPI cards ───────────────────────────── */
.kpi-card {
    background: rgba(22, 27, 34, 0.85);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 12px 16px;
    text-align: center;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 4px 28px rgba(0,0,0,0.6);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 110px;
    margin-bottom: 8px;
}
.kpi-card .val {
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin: 0;
}
.kpi-card .lbl {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 7px;
    color: #6e7681;
}
/* Neon per-card colours */
.neon-blue   { color: #00b4ff; text-shadow: 0 0 14px rgba(0,180,255,0.55); }
.neon-green  { color: #00ff9f; text-shadow: 0 0 14px rgba(0,255,159,0.55); }
.neon-orange { color: #ff9f00; text-shadow: 0 0 14px rgba(255,159,0,0.55); }
.neon-purple { color: #c77dff; text-shadow: 0 0 14px rgba(199,125,255,0.55); }
.neon-teal   { color: #00f0e0; text-shadow: 0 0 14px rgba(0,240,224,0.55); }
.neon-red    { color: #ff4d4d; text-shadow: 0 0 14px rgba(255,77,77,0.55); }

/* ── Section / sub headers ─────────────────────────────── */
.sec {
    font-size: 1.1rem;
    font-weight: 800;
    color: #e6edf3;
    border-left: 3px solid #00b4ff;
    padding: 5px 0 5px 12px;
    margin: 22px 0 12px;
}
.sub {
    font-size: 0.75rem;
    font-weight: 700;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 10px 0 5px;
}

/* ── Status / insight boxes ────────────────────────────── */
.ok-box  { background:#071f0c; border:1px solid #238636; border-radius:7px;
           padding:9px 13px; color:#3fb950; font-size:0.85rem; margin:4px 0; }
.err-box { background:#1a0505; border:1px solid #b91c1c; border-radius:7px;
           padding:9px 13px; color:#ff4d4d; font-size:0.85rem; margin:4px 0; }
.warn-box{ background:#1c1400; border:1px solid #9e6a03; border-radius:7px;
           padding:9px 13px; color:#d29922; font-size:0.85rem; margin:4px 0; }
.insight {
    background: rgba(22,27,34,0.9);
    border: 1px solid #1e2733;
    border-left: 4px solid #00b4ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.89rem;
    color: #c9d1d9;
    line-height: 1.65;
}

/* ── Misc ──────────────────────────────────────────────── */
.stDataFrame { border-radius: 8px; }
hr { border-color: #1e2733 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="📂 Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv("retail_sales_dataset.csv")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    df["Month_dt"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    for col in ["Quantity", "Price per Unit", "Total Amount", "Age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Gender"].fillna("Unknown", inplace=True)
    df["Product Category"].fillna("Unknown", inplace=True)
    df.fillna(df.select_dtypes(include="number").median(), inplace=True)
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error("❌ **retail_sales_dataset.csv** not found in the app folder.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PROPHET MODEL  — @st.cache_resource so the model survives reruns
# NOTE: The cutoff datetime is passed directly to a go.Scatter trace.
#       add_vline() is intentionally NOT used — it triggers a TypeError
#       when Plotly tries to sum pandas Timestamps internally.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🤖 Training Prophet model…")
def build_forecast(_df: pd.DataFrame, horizon: int = 90):
    daily = (
        _df.groupby("Date")["Total Amount"]
           .sum().reset_index()
           .rename(columns={"Date": "ds", "Total Amount": "y"})
    )
    if len(daily) < 2:
        return None, None
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        interval_width=0.90,
    )
    m.fit(daily)
    future   = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)
    return forecast, daily

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — filter defaults
# ══════════════════════════════════════════════════════════════════════════════
ALL_GENDERS = sorted(df_raw["Gender"].dropna().unique().tolist())
ALL_CATS    = sorted(df_raw["Product Category"].dropna().unique().tolist())
AGE_MIN     = int(df_raw["Age"].min())
AGE_MAX     = int(df_raw["Age"].max())
DATE_MIN    = df_raw["Date"].min().date()
DATE_MAX    = df_raw["Date"].max().date()

def _init_state():
    defaults = {
        "f_gender": ALL_GENDERS,
        "f_cat":    ALL_CATS,
        "f_age":    (AGE_MIN, AGE_MAX),
        "f_dates":  (DATE_MIN, DATE_MAX),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_filters():
    st.session_state["f_gender"] = ALL_GENDERS
    st.session_state["f_cat"]    = ALL_CATS
    st.session_state["f_age"]    = (AGE_MIN, AGE_MAX)
    st.session_state["f_dates"]  = (DATE_MIN, DATE_MAX)

_init_state()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Retail Intelligence")
    st.markdown("---")
    st.markdown("### Global Filters")

    sel_gender = st.multiselect(
        "Gender", ALL_GENDERS,
        default=st.session_state["f_gender"],
        key="f_gender",
    )
    sel_cat = st.multiselect(
        "Product Category", ALL_CATS,
        default=st.session_state["f_cat"],
        key="f_cat",
    )
    age_range = st.slider(
        "Age Range", AGE_MIN, AGE_MAX,
        value=st.session_state["f_age"],
        key="f_age",
    )
    date_range = st.date_input(
        "Date Range",
        value=st.session_state["f_dates"],
        min_value=DATE_MIN, max_value=DATE_MAX,
        key="f_dates",
    )

    st.markdown("")
    st.button("🔄 Reset Filters", on_click=reset_filters, use_container_width=True)

    st.markdown("---")
    st.caption("Filters apply across all tabs.")

# ══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ══════════════════════════════════════════════════════════════════════════════
df = df_raw.copy()
if sel_gender:
    df = df[df["Gender"].isin(sel_gender)]
if sel_cat:
    df = df[df["Product Category"].isin(sel_cat)]
df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
if len(date_range) == 2:
    df = df[
        (df["Date"].dt.date >= date_range[0]) &
        (df["Date"].dt.date <= date_range[1])
    ]

if df.empty:
    st.warning("⚠️ No records match the current filters. Click **Reset Filters** in the sidebar.")
    st.stop()

DR0 = date_range[0] if len(date_range) == 2 else DATE_MIN
DR1 = date_range[1] if len(date_range) == 2 else DATE_MAX

# ── Chart helpers ──────────────────────────────────────────────────────────────
DARK   = "plotly_dark"
TRANSP = "rgba(0,0,0,0)"
GMAP   = {"Male": "#00b4ff", "Female": "#c77dff", "Unknown": "#8b949e"}

def _dl(fig, h=380, **kw):
    """Apply dark transparent layout. Caller kwargs override defaults."""
    kw.setdefault("margin", dict(t=28, b=28, l=10, r=10))
    fig.update_layout(
        template=DARK, paper_bgcolor=TRANSP, plot_bgcolor=TRANSP,
        height=h, **kw,
    )
    return fig

def kpi(col, val: str, label: str, cls: str):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="val {cls}">{val}</div>'
        f'<div class="lbl">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def sec(txt: str, color: str = "#00b4ff"):
    st.markdown(
        f'<div class="sec" style="border-color:{color};">{txt}</div>',
        unsafe_allow_html=True,
    )

def sub(txt: str):
    st.markdown(f'<div class="sub">{txt}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='font-size:1.9rem;font-weight:900;color:#e6edf3;"
    "margin-bottom:2px;'>📊 Enterprise Retail Intelligence</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:#6e7681;font-size:0.86rem;margin-top:0;'>"
    f"<b style='color:#c9d1d9;'>{len(df):,}</b> of "
    f"<b style='color:#c9d1d9;'>{len(df_raw):,}</b> transactions &nbsp;·&nbsp; "
    f"<b style='color:#c9d1d9;'>{DR0}</b> → "
    f"<b style='color:#c9d1d9;'>{DR1}</b></p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Executive Overview",
    "👥 Demographic Deep-Dive",
    "📈 Strategic Forecasting",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # KPI metrics
    total_rev = df["Total Amount"].sum()
    avg_order = df["Total Amount"].mean()
    total_qty = df["Quantity"].sum()
    n_cust    = df["Customer ID"].nunique()
    avg_age   = df["Age"].mean()
    n_cat     = df["Product Category"].nunique()

    sec("📊 Key Performance Indicators", "#00b4ff")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi(k1, f"₹{total_rev:,.0f}",  "Total Revenue",       "neon-blue")
    kpi(k2, f"₹{avg_order:,.0f}",  "Avg Order Value",     "neon-green")
    kpi(k3, f"{total_qty:,}",       "Total Units Sold",    "neon-orange")
    kpi(k4, f"{n_cust:,}",          "Unique Customers",    "neon-purple")
    kpi(k5, f"{avg_age:.1f} yrs",   "Avg Customer Age",    "neon-teal")
    kpi(k6, f"{n_cat}",             "Product Categories",  "neon-red")

    st.markdown("<br>", unsafe_allow_html=True)

    # Time series
    sec("📈 Revenue Time Series", "#3fb950")
    monthly = (df.groupby("Month_dt")["Total Amount"]
                 .sum().reset_index().sort_values("Month_dt"))

    left, right = st.columns(2)

    with left:
        sub("Cumulative Revenue Over Time")
        cum = monthly.copy()
        cum["Cumulative"] = cum["Total Amount"].cumsum()
        f_cum = px.area(
            cum, x="Month_dt", y="Cumulative",
            labels={"Month_dt": "Month", "Cumulative": "Cumulative Revenue (₹)"},
            color_discrete_sequence=["#00b4ff"], template=DARK,
        )
        f_cum.update_traces(fillcolor="rgba(0,180,255,0.12)", line_color="#00b4ff")
        _dl(f_cum, h=360, hovermode="x unified",
            xaxis_title="Month", yaxis_title="Cumulative Revenue (₹)")
        st.plotly_chart(f_cum, use_container_width=True)

    with right:
        sub("Month-over-Month Revenue Growth (%)")
        mom = monthly.copy()
        mom["MoM %"] = (mom["Total Amount"].pct_change() * 100).round(2)
        mom["Label"] = mom["Month_dt"].dt.strftime("%b %Y")
        mom["Color"] = mom["MoM %"].apply(
            lambda x: "#00ff9f" if pd.notna(x) and x >= 0 else "#ff4d4d"
        )
        mom_c = mom.dropna(subset=["MoM %"])
        f_mom = go.Figure(go.Bar(
            x=mom_c["Label"], y=mom_c["MoM %"],
            marker_color=mom_c["Color"],
            text=mom_c["MoM %"].apply(lambda x: f"{x:+.1f}%"),
            textposition="outside",
            hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
        ))
        _dl(f_mom, h=360, xaxis_title="Month", yaxis_title="Growth (%)",
            yaxis=dict(zeroline=True, zerolinecolor="#00b4ff", zerolinewidth=1.5))
        st.plotly_chart(f_mom, use_container_width=True)

    # Data Quality expander
    with st.expander("🛠️ Diagnostic Data Quality Report", expanded=False):
        dq1, dq2 = st.columns(2)

        with dq1:
            sub("Missing Values per Column")
            miss = df_raw.isnull().sum().reset_index()
            miss.columns = ["Column", "Missing"]
            miss["Missing %"] = (miss["Missing"] / len(df_raw) * 100).round(2)
            st.dataframe(miss, use_container_width=True, hide_index=True)
            n_miss = miss["Missing"].sum()
            if n_miss == 0:
                st.markdown('<div class="ok-box">✅ No missing values in the dataset.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warn-box">⚠️ {n_miss} missing value(s) found.</div>',
                            unsafe_allow_html=True)

        with dq2:
            sub("Total Amount Integrity (Qty × Price per Unit)")
            chk = df_raw.copy()
            chk["Computed"] = chk["Quantity"] * chk["Price per Unit"]
            chk["OK"] = np.isclose(chk["Total Amount"], chk["Computed"], rtol=1e-3)
            n_bad = (~chk["OK"]).sum()
            pct   = chk["OK"].mean() * 100
            f_dq = go.Figure(go.Pie(
                labels=["Correct", "Mismatch"],
                values=[chk["OK"].sum(), n_bad],
                hole=0.62,
                marker_colors=["#00ff9f", "#ff4d4d"],
                textinfo="label+percent",
            ))
            f_dq.update_layout(
                template=DARK, paper_bgcolor=TRANSP, plot_bgcolor=TRANSP,
                height=200, showlegend=False,
                margin=dict(t=5, b=5, l=5, r=5),
                annotations=[dict(text=f"{pct:.1f}%<br>OK",
                                  x=0.5, y=0.5, font_size=13,
                                  showarrow=False, font_color="#e6edf3")],
            )
            st.plotly_chart(f_dq, use_container_width=True)
            if n_bad == 0:
                st.success("✅ All Total Amount values match Quantity × Price.")
            else:
                st.error(f"❌ {n_bad} row(s) have mismatched Total Amount.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEMOGRAPHIC DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Category ──────────────────────────────────────────────────────────────
    sec("🏷️ Product Category Analysis", "#d29922")
    cat = (df.groupby("Product Category")
             .agg(Revenue=("Total Amount", "sum"),
                  Units=("Quantity", "sum"),
                  Transactions=("Transaction ID", "count"),
                  Avg_Order=("Total Amount", "mean"),
                  Avg_Price=("Price per Unit", "mean"))
             .reset_index().sort_values("Revenue", ascending=False))

    c1, c2, c3 = st.columns(3)
    with c1:
        sub("Revenue by Category")
        fc1 = px.bar(cat, x="Product Category", y="Revenue",
                     color="Product Category", text_auto=".3s",
                     labels={"Revenue": "Revenue (₹)"},
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     template=DARK)
        fc1.update_traces(texttemplate="₹%{y:.3s}", textposition="outside")
        _dl(fc1, h=330, showlegend=False)
        st.plotly_chart(fc1, use_container_width=True)

    with c2:
        sub("Units Sold by Category")
        fc2 = px.bar(cat, x="Product Category", y="Units",
                     color="Product Category", text_auto=True,
                     labels={"Units": "Units Sold"},
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     template=DARK)
        fc2.update_traces(textposition="outside")
        _dl(fc2, h=330, showlegend=False)
        st.plotly_chart(fc2, use_container_width=True)

    with c3:
        sub("Revenue Share (Donut)")
        fc3 = px.pie(cat, names="Product Category", values="Revenue",
                     hole=0.46,
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     template=DARK)
        fc3.update_traces(texttemplate="%{label}<br>%{percent}")
        _dl(fc3, h=330, margin=dict(t=10, b=10, l=5, r=5))
        st.plotly_chart(fc3, use_container_width=True)

    sub("Category Summary Table")
    cat_d = cat.rename(columns={
        "Product Category": "Category",
        "Revenue": "Revenue (₹)",
        "Units": "Units Sold",
        "Avg_Order": "Avg Order (₹)",
        "Avg_Price": "Avg Price (₹)",
    })
    for c in ["Revenue (₹)", "Avg Order (₹)", "Avg Price (₹)"]:
        cat_d[c] = cat_d[c].map("₹{:,.2f}".format)
    st.dataframe(cat_d, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── Gender ─────────────────────────────────────────────────────────────────
    sec("👥 Gender Analysis", "#c77dff")
    gdf = (df.groupby("Gender")["Total Amount"]
             .agg(["sum", "count"]).reset_index())
    gdf.columns = ["Gender", "Revenue", "Transactions"]
    gdf["Avg Order"] = gdf["Revenue"] / gdf["Transactions"]

    g1, g2, g3 = st.columns(3)
    with g1:
        sub("Revenue Share by Gender")
        fg1 = px.pie(gdf, names="Gender", values="Revenue",
                     hole=0.46, color="Gender",
                     color_discrete_map=GMAP, template=DARK)
        fg1.update_traces(texttemplate="%{label}<br>₹%{value:,.0f}<br>%{percent}")
        _dl(fg1, h=290, margin=dict(t=10, b=10, l=5, r=5))
        st.plotly_chart(fg1, use_container_width=True)

    with g2:
        sub("Transaction Share by Gender")
        fg2 = px.pie(gdf, names="Gender", values="Transactions",
                     hole=0.46, color="Gender",
                     color_discrete_map=GMAP, template=DARK)
        fg2.update_traces(texttemplate="%{label}<br>%{value:,} txns<br>%{percent}")
        _dl(fg2, h=290, margin=dict(t=10, b=10, l=5, r=5))
        st.plotly_chart(fg2, use_container_width=True)

    with g3:
        sub("Avg Order Value by Gender")
        fg3 = px.bar(gdf, x="Gender", y="Avg Order",
                     color="Gender", text_auto=",.0f",
                     color_discrete_map=GMAP, template=DARK)
        fg3.update_traces(texttemplate="₹%{y:,.0f}", textposition="outside")
        _dl(fg3, h=290, showlegend=False)
        st.plotly_chart(fg3, use_container_width=True)

    st.markdown("---")

    # ── Age × Category heatmap ─────────────────────────────────────────────────
    sec("🎂 Age × Category Revenue Heatmap", "#00f0e0")
    df_hm = df.copy()
    df_hm["Age Group"] = pd.cut(
        df_hm["Age"],
        bins=[0, 17, 25, 35, 45, 55, 200],
        labels=["<18", "18-25", "26-35", "36-45", "46-55", "55+"],
        right=True,
    )
    pivot = (df_hm.groupby(["Age Group", "Product Category"], observed=True)
                  ["Total Amount"].sum().unstack(fill_value=0))
    fheat = px.imshow(
        pivot,
        labels=dict(color="Revenue (₹)"),
        color_continuous_scale="Blues",
        text_auto=".3s",
        aspect="auto",
        template=DARK,
    )
    _dl(fheat, h=310, margin=dict(t=15, b=15, l=15, r=15))
    st.plotly_chart(fheat, use_container_width=True)

    # Transaction table
    with st.expander("📋 View Filtered Transactions", expanded=False):
        srch = st.text_input("Search Customer ID or Category", "")
        show = df[
            df["Customer ID"].astype(str).str.contains(srch, case=False, na=False) |
            df["Product Category"].str.contains(srch, case=False, na=False)
        ] if srch else df
        cols = ["Transaction ID", "Date", "Customer ID", "Gender", "Age",
                "Product Category", "Quantity", "Price per Unit", "Total Amount"]
        st.dataframe(show[cols].sort_values("Date", ascending=False),
                     use_container_width=True, hide_index=True)
        st.caption(f"{len(show):,} rows shown")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STRATEGIC FORECASTING & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    # Pre-compute insight facts
    cat_s = (df.groupby("Product Category")
               .agg(Rev=("Total Amount", "sum"),
                    Txn=("Transaction ID", "count"),
                    Units=("Quantity", "sum"),
                    Avg=("Total Amount", "mean"))
               .reset_index().sort_values("Rev", ascending=False))

    df_a = df.copy()
    df_a["Age Group"] = pd.cut(
        df_a["Age"],
        bins=[0, 17, 25, 35, 45, 55, 200],
        labels=["<18", "18-25", "26-35", "36-45", "46-55", "55+"],
        right=True,
    )
    age_s = (df_a.groupby("Age Group", observed=True)["Total Amount"]
                 .agg(["sum", "count", "mean"]).reset_index())
    age_s.columns = ["Age Group", "Rev", "Txn", "Avg"]

    mth = (df.groupby("Month_dt")["Total Amount"]
             .sum().reset_index().sort_values("Month_dt"))

    top_cat    = cat_s.iloc[0]["Product Category"]
    top_rev    = cat_s.iloc[0]["Rev"]
    top_units  = cat_s.iloc[0]["Units"]
    top_txn    = cat_s.iloc[0]["Txn"]
    low_cat    = cat_s.iloc[-1]["Product Category"]
    low_rev    = cat_s.iloc[-1]["Rev"]
    low_txn    = cat_s.iloc[-1]["Txn"]
    top_gender = df.groupby("Gender")["Total Amount"].sum().idxmax()
    top_age    = age_s.sort_values("Rev", ascending=False).iloc[0]
    peak_row   = mth.sort_values("Total Amount", ascending=False).iloc[0]
    peak_str   = peak_row["Month_dt"].strftime("%B %Y")
    qty1_pct   = (df["Quantity"] == 1).mean() * 100
    avg_qty    = df["Quantity"].mean()

    # ── Forecast chart ─────────────────────────────────────────────────────────
    sec("📡 90-Day Revenue Forecast (Prophet)", "#00b4ff")

    forecast, daily = build_forecast(df, horizon=90)

    if forecast is None:
        st.warning("⚠️ Not enough data to build a forecast with the current filters.")
    else:
        cutoff = daily["ds"].max()  # pandas Timestamp

        # Forecast KPI cards (above chart)
        future_only = forecast[forecast["ds"] > cutoff]
        fc_total    = future_only["yhat"].clip(lower=0).sum()
        fc_daily    = future_only["yhat"].clip(lower=0).mean()
        fc_peak_day = (future_only.loc[future_only["yhat"].idxmax(), "ds"]
                       .strftime("%d %b %Y"))

        fk1, fk2, fk3 = st.columns(3)
        kpi(fk1, f"₹{fc_total:,.0f}",  "Projected 90-Day Revenue",     "neon-green")
        kpi(fk2, f"₹{fc_daily:,.0f}",  "Avg Forecasted Daily Revenue", "neon-blue")
        kpi(fk3, fc_peak_day,           "Predicted Peak Revenue Date",  "neon-orange")

        st.markdown("<br>", unsafe_allow_html=True)

        # Build chart
        fig_fc = go.Figure()

        # Historical line
        fig_fc.add_trace(go.Scatter(
            x=daily["ds"], y=daily["y"],
            mode="lines",
            name="Historical Revenue",
            line=dict(color="#00b4ff", width=1.8),
        ))

        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="#00ff9f", width=2.2, dash="dot"),
        ))

        # Confidence interval (shaded band)
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"].iloc[::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(0,255,159,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="90% Confidence Interval",
        ))

        # ── Vertical "Forecast Start" line ────────────────────────────────────
        # IMPORTANT: We draw this as a go.Scatter trace spanning the full y-range.
        # We do NOT use fig.add_vline(x=...) because Plotly internally calls
        # _mean([cutoff, cutoff]) which tries to sum two Timestamps, throwing:
        #   TypeError: unsupported operand type(s) for +: 'int' and 'str'
        # Passing the raw pd.Timestamp as x-coordinates to a Scatter trace
        # is handled correctly by Plotly without any arithmetic on the values.
        y_lo = min(forecast["yhat_lower"].min(), daily["y"].min()) * 0.95
        y_hi = forecast["yhat_upper"].max() * 1.05
        fig_fc.add_trace(go.Scatter(
            x=[cutoff, cutoff],
            y=[y_lo, y_hi],
            mode="lines",
            name="Forecast Start",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1.5),
            hoverinfo="skip",
            showlegend=True,
        ))

        _dl(fig_fc, h=460,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Daily Revenue (₹)",
            legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")

    # ── Prescriptive Actions table ─────────────────────────────────────────────
    sec("📋 Decision Support Matrix", "#c77dff")

    actions = pd.DataFrame([
        {
            "Business Area":      "Inventory",
            "Observation":        f"'{top_cat}' drives the highest revenue — ₹{top_rev:,.0f} ({top_units:,} units)",
            "Recommended Action": f"Increase {top_cat} stock 20–30%; introduce premium SKUs to raise avg selling price",
            "Priority":           "🔴 High",
            "Timeline":           "0–30 days",
        },
        {
            "Business Area":      "Underperformer",
            "Observation":        f"'{low_cat}' — only ₹{low_rev:,.0f} from {low_txn} transactions",
            "Recommended Action": f"30-day markdown trial + bundle {low_cat} with {top_cat}; set 90-day revenue gate",
            "Priority":           "🟡 Medium",
            "Timeline":           "30–60 days",
        },
        {
            "Business Area":      "Marketing",
            "Observation":        f"'{top_gender}' customers generate the highest total revenue",
            "Recommended Action": f"Shift 30% of ad budget to {top_gender}-skewed channels; A/B test creatives",
            "Priority":           "🔴 High",
            "Timeline":           "0–30 days",
        },
        {
            "Business Area":      "Retention",
            "Observation":        f"Age group '{top_age['Age Group']}' is the highest-value segment — ₹{top_age['Rev']:,.0f}",
            "Recommended Action": f"Launch tiered loyalty programme for {top_age['Age Group']}; personalised email series",
            "Priority":           "🔴 High",
            "Timeline":           "30–90 days",
        },
        {
            "Business Area":      "AOV Growth",
            "Observation":        f"{qty1_pct:.1f}% of orders are single-unit; avg cart = {avg_qty:.2f} units",
            "Recommended Action": "'Buy 2 Save 10%' bundles; free-shipping threshold; 'Frequently Bought Together'",
            "Priority":           "🟡 Medium",
            "Timeline":           "14–30 days",
        },
        {
            "Business Area":      "Seasonality",
            "Observation":        f"Peak revenue month: {peak_str}",
            "Recommended Action": "Pre-stock 6–8 weeks ahead; schedule paid media 4 weeks before peak",
            "Priority":           "🟢 Low",
            "Timeline":           "60–90 days",
        },
    ])
    st.dataframe(actions, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Diagnostic insight bullets ─────────────────────────────────────────────
    sec("💡 Diagnostic Insights", "#d29922")

    insights = [
        (f"🏆 <b>{top_cat}</b> is the clear revenue champion at "
         f"<b>₹{top_rev:,.0f}</b> across <b>{top_txn:,}</b> transactions. "
         "Sustained inventory investment and promotional spend here will compound returns."),

        (f"📉 <b>{low_cat}</b> is significantly underperforming "
         f"(₹{low_rev:,.0f}, {low_txn} txns). A pricing audit, markdown trial, "
         f"and cross-sell pairing with {top_cat} should be launched within 30 days."),

        (f"👤 <b>{top_gender}</b> customers are the dominant revenue cohort. "
         "Channel-specific creative testing and segment-targeted promotions can "
         "deepen this advantage further."),

        (f"🎯 The <b>{top_age['Age Group']}</b> age bracket generates "
         f"<b>₹{top_age['Rev']:,.0f}</b> with an avg order of "
         f"<b>₹{top_age['Avg']:,.0f}</b>. A tailored loyalty programme here "
         "will yield the highest lifetime-value ROI."),

        (f"🛒 <b>{qty1_pct:.0f}%</b> of all orders are single-unit purchases "
         f"(avg cart: {avg_qty:.2f} units). Bundle mechanics and volume discounts "
         "could lift Average Order Value by 15–20% within one quarter."),
    ]

    for ins in insights:
        st.markdown(f'<div class="insight">{ins}</div>', unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#3d444d;font-size:0.72rem;'>"
    "Enterprise Retail Intelligence · Streamlit · Plotly · Prophet"
    "</p>",
    unsafe_allow_html=True,
)
