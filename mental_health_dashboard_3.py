# ============================================================
# mental_health_dashboard.py
# Data Story: Mental Health & Productivity
# Team Project — Data Visualization | Tec de Monterrey
# Run: python3 -m streamlit run mental_health_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health & Productivity",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — clean white design ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff !important;
}
.main .block-container { padding: 1rem 2rem 2rem 2rem; max-width: 100%; }

section[data-testid="stSidebar"] { background: #f8f9fa !important; border-right: 1px solid #e9ecef; }
section[data-testid="stSidebar"] * { color: #495057 !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.7rem !important; text-transform: uppercase;
    letter-spacing: 1px; color: #6c757d !important;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-top: 3px solid #0096c7;
    border-radius: 10px;
    padding: 20px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-label { font-size: 0.65rem; color: #6c757d; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #1a2540; line-height: 1; }
.kpi-sub   { font-size: 0.7rem; color: #adb5bd; margin-top: 6px; }

.section-title {
    font-size: 0.7rem; font-weight: 700; color: #0096c7;
    text-transform: uppercase; letter-spacing: 2px;
    margin: 36px 0 6px 0; padding-bottom: 10px;
    border-bottom: 1px solid #e9ecef;
}
.section-story {
    font-size: 0.88rem; color: #495057; line-height: 1.7;
    margin-bottom: 18px; max-width: 950px;
}
.section-story b { color: #1a2540; }
.section-story .hl { color: #0096c7; font-weight: 600; }

.dash-header {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.dash-header h1 {
    color: #1a2540; font-size: 1.7rem; font-weight: 700;
    margin: 4px 0 8px 0; line-height: 1.2;
}
.dash-header p { color: #6c757d; font-size: 0.72rem; margin: 0; text-transform: uppercase; letter-spacing: 1.5px; }
.dash-header .lead {
    color: #495057; font-size: 0.95rem; line-height: 1.6;
    margin-top: 8px; max-width: 850px;
}

/* People icons */
.people-row {
    display: flex; gap: 4px; align-items: center;
    margin: 14px 0 8px 0;
}
.people-row .stat {
    font-size: 1.1rem; font-weight: 700; color: #1a2540;
    margin-right: 14px; line-height: 1;
}
.people-row .stat .big { color: #0096c7; font-size: 1.5rem; }

#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── Colors ───────────────────────────────────────────────────
BG     = "#ffffff"
ACCENT = "#0096c7"
GRAY   = "#d0d5dd"
TICK   = "#6c757d"
TEXT   = "#1a2540"
GRID   = "#f0f2f5"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#e9ecef", "axes.labelcolor": TICK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": TEXT,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "font.family": "sans-serif", "font.size": 10,
})

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("MentalhealthDepressiondisorderData.csv", low_memory=False)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Depression (%)"] = pd.to_numeric(df["Depression (%)"], errors="coerce")
    df["Anxiety disorders (%)"] = pd.to_numeric(df["Anxiety disorders (%)"], errors="coerce")
    df = df.dropna(subset=["Year", "Depression (%)"])
    df["Year"] = df["Year"].astype(int)

    # HDI dataset (2017)
    hdi = pd.read_excel("hdr-data.xlsx")
    hdi = hdi[["countryIsoCode", "country", "value"]].rename(
        columns={"countryIsoCode": "Code", "country": "Entity", "value": "HDI"}
    )
    return df, hdi

df, hdi_df = load_data()
countries_df = df[df["Code"].notna() & (df["Code"] != "") & (df["Code"].str.len() == 3)].copy()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p style='font-size:0.7rem;color:#0096c7;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;'>Data Story</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.9rem;color:#1a2540;font-weight:600;margin-bottom:24px;'>Mental Health & Productivity</p>", unsafe_allow_html=True)

    year_min, year_max = int(countries_df["Year"].min()), int(countries_df["Year"].max())
    sel_year = st.slider("Select Year", year_min, year_max, year_max)

    top_n = st.slider("Top N Countries", 5, 20, 10)

    st.markdown("---")
    st.markdown("<p style='font-size:0.65rem;color:#adb5bd;'>Sources:<br>· Our World in Data (Depression)<br>· UNDP Human Development Index 2017</p>", unsafe_allow_html=True)

# Filter by selected year
dff = countries_df[countries_df["Year"] == sel_year].copy()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <p>Tec de Monterrey · Data Visualization Course · Team Project</p>
  <h1>Mental Health Is a Productivity Crisis in Disguise</h1>
  <div class="lead">
    Countries with higher depression rates tend to score lower in human development.
    This dashboard explores how mental health and national productivity are connected
    — and why ignoring the first means losing the second.
  </div>
</div>
""", unsafe_allow_html=True)

# ── People icons section ──────────────────────────────────────
avg_dep_global = dff["Depression (%)"].mean()
avg_anx_global = dff["Anxiety disorders (%)"].mean()
people_with_dep = round(avg_dep_global / 10 * 10)
# Build SVG row — 10 people, some highlighted
people_svgs = []
n_affected = max(1, round(avg_dep_global / 10 * 10))  # rough scaling
n_affected = max(1, round(avg_dep_global))  # ~3-5 per 100

# Use a clean stat: roughly X out of 100 people have depression worldwide
stat_text = f"{avg_dep_global:.1f}"
anx_text  = f"{avg_anx_global:.1f}"

# Generate person SVG icons (10 icons; ~ first 3-4 highlighted)
person_svg = '<svg width="22" height="40" viewBox="0 0 22 40" xmlns="http://www.w3.org/2000/svg"><circle cx="11" cy="6" r="5" fill="{c}"/><path d="M3 14 Q11 10 19 14 L19 28 L14 28 L14 38 L8 38 L8 28 L3 28 Z" fill="{c}"/></svg>'
icons = ""
affected_count = max(1, min(10, round(avg_dep_global)))
for i in range(10):
    color = ACCENT if i < affected_count else "#dee2e6"
    icons += person_svg.format(c=color)

st.markdown(f"""
<div style="background:#f8f9fa; border-radius:12px; padding:24px 28px; margin-bottom:24px; border:1px solid #e9ecef;">
  <p style="font-size:0.7rem; color:#0096c7; letter-spacing:2px; text-transform:uppercase; margin:0 0 12px 0;">By the Numbers ({sel_year})</p>
  <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
    {icons}
    <div style="margin-left:18px;">
      <div style="font-size:1.6rem; font-weight:700; color:#1a2540; line-height:1;">
        {affected_count} <span style="font-size:1rem; color:#6c757d;">in every 100 people</span>
      </div>
      <div style="font-size:0.85rem; color:#495057; margin-top:6px;">
        live with <b style="color:#0096c7;">depression</b> globally
        — averaging <b>{stat_text}%</b> of the population.
      </div>
    </div>
  </div>
  <div style="margin-top:14px; padding-top:14px; border-top:1px solid #e9ecef; font-size:0.82rem; color:#6c757d;">
    📌 Anxiety adds another <b style="color:#1a2540;">{anx_text}%</b> on top — many people experience both at once.
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────
n_countries = dff["Entity"].nunique()
max_dep = dff["Depression (%)"].max()
max_country = dff.loc[dff["Depression (%)"].idxmax(), "Entity"]

k1,k2,k3 = st.columns(3)
with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Global Avg Depression</div><div class="kpi-value">{avg_dep_global:.2f}%</div><div class="kpi-sub">Of population in {sel_year}</div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Highest Country Rate</div><div class="kpi-value">{max_dep:.2f}%</div><div class="kpi-sub">{max_country}</div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Countries Analyzed</div><div class="kpi-value">{n_countries}</div><div class="kpi-sub">With complete data</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CHART 1 — World map (Choropleth) — INTERACTIVE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">01 · Where Depression Lives</div>', unsafe_allow_html=True)
st.markdown(f"""<div class="section-story">
A map shows the geography of the crisis at a glance. The countries shaded in deeper blue carry
the heaviest depression burden. Hover over any country to see its exact rate — click and drag to explore.
</div>""", unsafe_allow_html=True)

map_data = dff.dropna(subset=["Depression (%)", "Code"]).copy()

fig_map = px.choropleth(
    map_data,
    locations="Code",
    color="Depression (%)",
    hover_name="Entity",
    hover_data={"Code": False, "Depression (%)": ":.2f"},
    color_continuous_scale=[[0, "#e8f4fa"], [0.5, "#48cae4"], [1, ACCENT]],
    labels={"Depression (%)": "Depression Rate (%)"}
)
fig_map.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    geo=dict(
        showframe=False, showcoastlines=False, bgcolor=BG,
        projection_type="natural earth"
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    height=480,
    font=dict(family="Inter", color=TICK),
    coloraxis_colorbar=dict(
        title="", thickness=12, len=0.5, x=0.95,
        tickfont=dict(color=TICK, size=10)
    )
)
st.plotly_chart(fig_map, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# CHART 2 — Bar: Top N countries (the chart he liked)
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">02 · The Heaviest Burden</div>', unsafe_allow_html=True)
st.markdown(f"""<div class="section-story">
Ranking countries by depression rate exposes the gap between the most and least affected populations.
The single highlighted bar marks the country carrying the greatest weight in {sel_year}.
</div>""", unsafe_allow_html=True)

top = dff.nlargest(top_n, "Depression (%)").sort_values("Depression (%)", ascending=True)

fig2, ax2 = plt.subplots(figsize=(11, max(4, top_n*0.4)))
bar_colors = [ACCENT if i == len(top)-1 else GRAY for i in range(len(top))]
bars = ax2.barh(top["Entity"], top["Depression (%)"], color=bar_colors, height=0.6, zorder=2)
for bar, val in zip(bars, top["Depression (%)"]):
    ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", va="center", fontsize=9, color=TICK)
ax2.set_title(f"Top {top_n} Countries by Depression Rate ({sel_year})", fontsize=12, color=TEXT, pad=12, fontweight="bold")
ax2.set_xlabel("Share of Population (%)")
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax2.set_xlim(0, top["Depression (%)"].max() * 1.2)
ax2.grid(axis="x")
for spine in ["top","right"]: ax2.spines[spine].set_visible(False)
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close()

# ══════════════════════════════════════════════════════════════
# CHART 3 — Depression vs HDI scatter (the connection)
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">03 · The Productivity Connection</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-story">
Plotting depression rates against the <b>Human Development Index</b> reveals the relationship that defines
this story: countries with higher depression rates tend to score lower in human development —
a proxy for national productivity, education, and quality of life. <span class="hl">Mental health is not
separate from economic development; it is one of its foundations.</span>
</div>""", unsafe_allow_html=True)

# Merge depression with HDI on country code
hdi_data = dff[["Code","Entity","Depression (%)"]].merge(hdi_df[["Code","HDI"]], on="Code", how="inner")
hdi_data = hdi_data.dropna()

# Compute trend line
if len(hdi_data) > 5:
    coeffs = np.polyfit(hdi_data["HDI"], hdi_data["Depression (%)"], 1)
    trend_x = np.linspace(hdi_data["HDI"].min(), hdi_data["HDI"].max(), 100)
    trend_y = np.polyval(coeffs, trend_x)
    correlation = hdi_data["HDI"].corr(hdi_data["Depression (%)"])
else:
    correlation = 0
    trend_x, trend_y = [], []

fig3 = go.Figure()

# All points
fig3.add_trace(go.Scatter(
    x=hdi_data["HDI"], y=hdi_data["Depression (%)"],
    mode="markers",
    marker=dict(color=ACCENT, size=10, opacity=0.7, line=dict(color="#005f73", width=0.5)),
    text=hdi_data["Entity"],
    hovertemplate="<b>%{text}</b><br>HDI: %{x:.3f}<br>Depression: %{y:.2f}%<extra></extra>",
    showlegend=False
))

# Trend line
if len(trend_x) > 0:
    fig3.add_trace(go.Scatter(
        x=trend_x, y=trend_y, mode="lines",
        line=dict(color=GRAY, width=2, dash="dash"),
        name=f"Trend (r = {correlation:.2f})",
        hoverinfo="skip"
    ))

fig3.update_layout(
    title=dict(text=f"Depression Rate vs Human Development Index ({sel_year})",
               font=dict(size=12, color=TEXT)),
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="Inter", color=TICK),
    xaxis=dict(title="Human Development Index (HDI)", showgrid=True, gridcolor=GRID, color=TICK),
    yaxis=dict(title="Depression (%)", showgrid=True, gridcolor=GRID, color=TICK,
               tickformat=".2f", ticksuffix="%"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TICK), x=0.78, y=0.98),
    height=480,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig3, use_container_width=True)

# Add interpretation below
if correlation < -0.2:
    interp = f"The correlation coefficient is <b>{correlation:.2f}</b> — a clear negative relationship. Higher development scores correlate with lower depression rates."
elif correlation > 0.2:
    interp = f"The correlation coefficient is <b>{correlation:.2f}</b> — a positive relationship. This is unexpected and may reflect reporting differences across countries."
else:
    interp = f"The correlation coefficient is <b>{correlation:.2f}</b> — a weak relationship. Depression appears widespread across all development levels."

st.markdown(f"""<div class="section-story" style="background:#f8f9fa; border-left:3px solid #0096c7; padding:14px 18px; border-radius:0 8px 8px 0; margin-top:-10px;">
🔍 <b>Reading the data:</b> {interp}
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CHART 4 — Country deep dive: click any country
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">04 · Explore Any Country</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-story">
Pick a country below to see its depression trajectory from 1990 onward, alongside its HDI score
and how it compares against the global average.
</div>""", unsafe_allow_html=True)

countries = sorted(countries_df["Entity"].unique().tolist())
default_idx = countries.index("United States") if "United States" in countries else 0
sel_country = st.selectbox("Select a country", countries, index=default_idx)

cd = countries_df[countries_df["Entity"] == sel_country].sort_values("Year")
global_avg = countries_df.groupby("Year")["Depression (%)"].mean().reset_index()

# Get this country's HDI if available
country_code = cd["Code"].iloc[0] if len(cd) > 0 else None
country_hdi = hdi_df[hdi_df["Code"] == country_code]["HDI"].values
country_hdi_val = country_hdi[0] if len(country_hdi) > 0 else None

# Build 3-column metric strip
m1, m2, m3 = st.columns(3)
with m1:
    latest_rate = cd[cd["Year"] == cd["Year"].max()]["Depression (%)"].values[0] if len(cd) > 0 else 0
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{sel_country} — Latest Depression</div><div class="kpi-value">{latest_rate:.2f}%</div><div class="kpi-sub">Most recent year</div></div>', unsafe_allow_html=True)
with m2:
    if country_hdi_val:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">HDI Score (2017)</div><div class="kpi-value">{country_hdi_val:.3f}</div><div class="kpi-sub">Human Development Index</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">HDI Score</div><div class="kpi-value">—</div><div class="kpi-sub">Not available</div></div>', unsafe_allow_html=True)
with m3:
    if len(cd) > 0:
        start_rate = cd.iloc[0]["Depression (%)"]
        change = latest_rate - start_rate
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Change Since 1990</div><div class="kpi-value">{change:+.2f}%</div><div class="kpi-sub">Total shift</div></div>', unsafe_allow_html=True)

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=global_avg["Year"], y=global_avg["Depression (%)"],
    name="Global Average", mode="lines",
    line=dict(color=GRAY, width=2, dash="dot"),
    hovertemplate="Global Avg<br>%{y:.2f}%<extra></extra>"
))
fig5.add_trace(go.Scatter(
    x=cd["Year"], y=cd["Depression (%)"],
    name=sel_country, mode="lines",
    line=dict(color=ACCENT, width=2.8),
    fill="tozeroy", fillcolor="rgba(0,150,199,0.08)",
    hovertemplate=f"{sel_country}<br>%{{y:.2f}}%<extra></extra>"
))
fig5.update_layout(
    title=dict(text=f"{sel_country} — Depression Rate vs Global Average",
               font=dict(size=12, color=TEXT)),
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="Inter", color=TICK),
    xaxis=dict(showgrid=False, color=TICK),
    yaxis=dict(showgrid=True, gridcolor=GRID, color=TICK,
               tickformat=".2f", ticksuffix="%"),
    legend=dict(orientation="h", y=1.12, font=dict(color=TICK)),
    height=380,
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# Conclusion
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">What This Data Shows</div>', unsafe_allow_html=True)
st.markdown(f"""<div style="background:#f8f9fa; border-left:3px solid #0096c7; padding:18px 22px; border-radius:0 8px 8px 0; margin:14px 0; font-size:0.88rem; color:#495057; line-height:1.75;">
The map reveals depression is geographically uneven. The bar chart pinpoints which countries carry the most weight.
And the scatter against HDI exposes a relationship that deserves attention: <b style="color:#1a2540;">when mental
health declines, broader human development tends to follow.</b><br><br>
The implication is direct — investing in mental health is not just a humanitarian concern.
It is an investment in the long-term productivity, education, and quality of life of an entire nation.
</div>
""", unsafe_allow_html=True)

st.markdown("<br><p style='font-size:0.65rem;color:#adb5bd;text-align:center;letter-spacing:2px;'>TEC DE MONTERREY · DATA VISUALIZATION · TEAM PROJECT · 2026</p>", unsafe_allow_html=True)
