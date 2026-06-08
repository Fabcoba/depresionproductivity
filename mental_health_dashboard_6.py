# ============================================================
# mental_health_dashboard.py
# Data Story: The Diagnosis Paradigm
# Run: python3 -m streamlit run mental_health_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Mental Health: The Diagnosis Paradigm",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
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
.kpi-card.green { border-top-color: #2d8659; }
.kpi-label { font-size: 0.65rem; color: #6c757d; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #1a2540; line-height: 1; }
.kpi-sub   { font-size: 0.7rem; color: #adb5bd; margin-top: 6px; }

.section-title {
    font-size: 0.7rem; font-weight: 700; color: #0096c7;
    text-transform: uppercase; letter-spacing: 2px;
    margin: 36px 0 6px 0; padding-bottom: 10px;
    border-bottom: 1px solid #e9ecef;
}
.section-title.green { color: #2d8659; }
.section-story {
    font-size: 0.88rem; color: #495057; line-height: 1.7;
    margin-bottom: 18px; max-width: 950px;
}
.section-story b { color: #1a2540; }

.dash-header {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.dash-header h1 {
    color: #1a2540; font-size: 1.7rem; font-weight: 700;
    margin: 0 0 8px 0; line-height: 1.2;
}
.dash-header .lead {
    color: #495057; font-size: 0.95rem; line-height: 1.6;
    margin-top: 8px; max-width: 850px;
}

#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# Colors
BG     = "#ffffff"
ACCENT = "#0096c7"
GREEN  = "#2d8659"
GREEN_LIGHT = "#d4ebde"
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

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("MentalhealthDepressiondisorderData.csv", low_memory=False)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Depression (%)"] = pd.to_numeric(df["Depression (%)"], errors="coerce")
    df["Anxiety disorders (%)"] = pd.to_numeric(df["Anxiety disorders (%)"], errors="coerce")
    df = df.dropna(subset=["Year", "Depression (%)"])
    df["Year"] = df["Year"].astype(int)

    health = pd.read_csv("annualhealthcareexpenditurepercapita.csv")
    health.columns = ["Entity", "Code", "Year", "HealthSpending"]
    health = health.dropna(subset=["HealthSpending", "Code"])

    return df, health

df, health_df = load_data()
countries_df = df[df["Code"].notna() & (df["Code"] != "") & (df["Code"].str.len() == 3)].copy()

# Find year that has both datasets
common_years = sorted(set(countries_df["Year"].unique()) & set(health_df["Year"].unique()))

# Sidebar
with st.sidebar:
    st.markdown("<p style='font-size:0.7rem;color:#0096c7;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;'>Data Story</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.9rem;color:#1a2540;font-weight:600;margin-bottom:24px;'>The Diagnosis Paradigm</p>", unsafe_allow_html=True)

    sel_year = st.select_slider("Select Year",
                                 options=common_years,
                                 value=max(common_years))

    top_n = st.slider("Top N Countries", 5, 20, 10)

    st.markdown("---")
    st.markdown("<p style='font-size:0.65rem;color:#adb5bd;'>Sources: Our World in Data, WHO Global Health Expenditure Database 2024</p>", unsafe_allow_html=True)

dff = countries_df[countries_df["Year"] == sel_year].copy()
hff = health_df[health_df["Year"] == sel_year].copy()

# Header
st.markdown("""
<div class="dash-header">
  <h1>The Diagnosis Paradigm: Where Depression Hides</h1>
  <div class="lead">
    Countries that invest more in healthcare report higher depression rates.
    This is not because their populations are sicker. It is because they diagnose what others ignore.
    The real burden of mental health is invisible in countries that cannot afford to count it.
  </div>
</div>
""", unsafe_allow_html=True)

# Lifetime prevalence: 2 in 10
person_svg = '<svg width="22" height="40" viewBox="0 0 22 40" xmlns="http://www.w3.org/2000/svg"><circle cx="11" cy="6" r="5" fill="{c}"/><path d="M3 14 Q11 10 19 14 L19 28 L14 28 L14 38 L8 38 L8 28 L3 28 Z" fill="{c}"/></svg>'
icons = ""
for i in range(10):
    color = ACCENT if i < 2 else "#dee2e6"
    icons += person_svg.format(c=color)

avg_dep_global = dff["Depression (%)"].mean()
avg_anx_global = dff["Anxiety disorders (%)"].mean()

st.markdown(f"""
<div style="background:#f8f9fa; border-radius:12px; padding:24px 28px; margin-bottom:24px; border:1px solid #e9ecef;">
  <p style="font-size:0.7rem; color:#0096c7; letter-spacing:2px; text-transform:uppercase; margin:0 0 12px 0;">The Human Scale</p>
  <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
    {icons}
    <div style="margin-left:18px;">
      <div style="font-size:1.6rem; font-weight:700; color:#1a2540; line-height:1;">
        2 <span style="font-size:1rem; color:#6c757d;">in every 10 people</span>
      </div>
      <div style="font-size:0.85rem; color:#495057; margin-top:6px;">
        have suffered or will suffer from <b style="color:#0096c7;">depression</b> at some point in their lives,
        according to WHO estimates.
      </div>
    </div>
  </div>
  <div style="margin-top:14px; padding-top:14px; border-top:1px solid #e9ecef; font-size:0.82rem; color:#6c757d;">
    In {sel_year}, the global reported depression rate was <b style="color:#1a2540;">{avg_dep_global:.1f}%</b>,
    but actual prevalence is closer to <b style="color:#1a2540;">20%</b>. The gap is what countries fail to diagnose.
  </div>
</div>
""", unsafe_allow_html=True)

# KPIs
n_countries = dff["Entity"].nunique()
max_dep = dff["Depression (%)"].max()
max_country = dff.loc[dff["Depression (%)"].idxmax(), "Entity"]
avg_health_spending = hff["HealthSpending"].mean() if len(hff) > 0 else 0
max_health = hff.loc[hff["HealthSpending"].idxmax(), "Entity"] if len(hff) > 0 else "N/A"

k1,k2,k3,k4 = st.columns(4)
with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Reported Depression</div><div class="kpi-value">{avg_dep_global:.2f}%</div><div class="kpi-sub">Global avg ({sel_year})</div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Highest Reported Rate</div><div class="kpi-value">{max_dep:.2f}%</div><div class="kpi-sub">{max_country}</div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Avg Health Spending</div><div class="kpi-value">${avg_health_spending:,.0f}</div><div class="kpi-sub">Per capita (PPP)</div></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Top Investor</div><div class="kpi-value" style="font-size:1.4rem;">{max_health[:18]}</div><div class="kpi-sub">Highest spending country</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MAP 1 — Depression (BLUE)
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">01 Where Depression Is Reported</div>', unsafe_allow_html=True)
st.markdown(f"""<div class="section-story">
Darker blue marks countries with higher reported depression rates. The pattern looks counterintuitive:
many of the highest rates appear in wealthy, developed nations. Hold that thought.
</div>""", unsafe_allow_html=True)

map_data = dff.dropna(subset=["Depression (%)", "Code"]).copy()
fig_map = px.choropleth(
    map_data, locations="Code", color="Depression (%)",
    hover_name="Entity", hover_data={"Code": False, "Depression (%)": ":.2f"},
    color_continuous_scale=[[0, "#e8f4fa"], [0.5, "#48cae4"], [1, ACCENT]],
    labels={"Depression (%)": "Depression Rate (%)"}
)
fig_map.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    geo=dict(showframe=False, showcoastlines=False, bgcolor=BG, projection_type="natural earth"),
    margin=dict(l=0, r=0, t=10, b=0), height=460,
    font=dict(family="Inter", color=TICK),
    coloraxis_colorbar=dict(title="", thickness=12, len=0.5, x=0.95, tickfont=dict(color=TICK, size=10))
)
st.plotly_chart(fig_map, use_container_width=True)

# ════════════════════════════════════════════════════════════
# MAP 2 — Healthcare Spending (GREEN)
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title green">02 Where Healthcare Is Funded</div>', unsafe_allow_html=True)
st.markdown(f"""<div class="section-story">
Now compare. Darker green marks countries that spend more per capita on healthcare.
The geographic overlap with the depression map is striking. Countries with strong health systems
report more depression, not less, because they have the infrastructure to detect it.
</div>""", unsafe_allow_html=True)

map_health_data = hff.dropna(subset=["HealthSpending", "Code"]).copy()
# Log transform for better color distribution
map_health_data["LogSpending"] = np.log10(map_health_data["HealthSpending"])

fig_map_h = px.choropleth(
    map_health_data, locations="Code", color="HealthSpending",
    hover_name="Entity",
    hover_data={"Code": False, "HealthSpending": ":,.0f", "LogSpending": False},
    color_continuous_scale=[[0, "#f0f8f3"], [0.3, "#b3dcc1"], [0.6, "#5fa87f"], [1, "#1a5c3a"]],
    labels={"HealthSpending": "Per Capita ($)"}
)
fig_map_h.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    geo=dict(showframe=False, showcoastlines=False, bgcolor=BG, projection_type="natural earth"),
    margin=dict(l=0, r=0, t=10, b=0), height=460,
    font=dict(family="Inter", color=TICK),
    coloraxis_colorbar=dict(title="USD PPP", thickness=12, len=0.5, x=0.95,
                              tickfont=dict(color=TICK, size=10),
                              titlefont=dict(color=TICK, size=10))
)
st.plotly_chart(fig_map_h, use_container_width=True)

# ════════════════════════════════════════════════════════════
# CHART 3 — Bar chart by spending quartile (THE KILLER CHART)
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">03 The Paradigm in One Chart</div>', unsafe_allow_html=True)
st.markdown(f"""<div class="section-story">
Grouping countries into quartiles by healthcare spending and averaging their reported depression rates
reveals the paradox. Low and high spending countries report similar rates, but for opposite reasons.
Low spending countries underdetect. High spending countries see what is actually there.
</div>""", unsafe_allow_html=True)

combined = dff[["Code","Entity","Depression (%)"]].merge(
    hff[["Code","HealthSpending"]], on="Code", how="inner"
).dropna()

if len(combined) > 20:
    combined["SpendingGroup"] = pd.qcut(combined["HealthSpending"], 4,
        labels=["Q1: Low Spending","Q2: Medium-Low","Q3: Medium-High","Q4: High Spending"])

    group_stats = combined.groupby("SpendingGroup", observed=True).agg(
        avg_dep=("Depression (%)", "mean"),
        avg_spending=("HealthSpending", "mean"),
        n_countries=("Entity", "count")
    ).reset_index()

    # Bar chart with green gradient
    fig_bar, ax_bar = plt.subplots(figsize=(11, 5))
    greens = ["#b3dcc1", "#7fbf9a", "#4a9e6f", "#1a5c3a"]
    bars = ax_bar.bar(group_stats["SpendingGroup"], group_stats["avg_dep"],
                      color=greens, width=0.6, zorder=2)

    # Add value labels on bars
    for bar, val, n in zip(bars, group_stats["avg_dep"], group_stats["n_countries"]):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                    f"{val:.2f}%", ha="center", fontsize=11, color=TEXT, fontweight="700")
        ax_bar.text(bar.get_x() + bar.get_width()/2, 0.15,
                    f"{n} countries", ha="center", fontsize=8, color="white", fontweight="600")

    # Add average spending below the labels
    for bar, spending in zip(bars, group_stats["avg_spending"]):
        ax_bar.text(bar.get_x() + bar.get_width()/2, -0.25,
                    f"avg ${spending:,.0f}", ha="center", fontsize=8.5, color=TICK, style="italic")

    ax_bar.set_title("Average Reported Depression Rate by Healthcare Spending Group",
                     fontsize=12, color=TEXT, pad=14, fontweight="bold")
    ax_bar.set_ylabel("Reported Depression (%)")
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax_bar.set_ylim(0, max(group_stats["avg_dep"]) * 1.3)
    ax_bar.grid(axis="y")
    for spine in ["top","right"]: ax_bar.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()

    # The key insight callout
    diff_q3_q1 = group_stats[group_stats["SpendingGroup"]=="Q3: Medium-High"]["avg_dep"].values[0] - group_stats[group_stats["SpendingGroup"]=="Q1: Low Spending"]["avg_dep"].values[0]

    st.markdown(f"""<div class="section-story" style="background:#f4f9f6; border-left:3px solid #2d8659; padding:16px 20px; border-radius:0 8px 8px 0;">
    <b>The paradox visualized.</b> Countries in Q1 (low spending) report <b>{group_stats.iloc[0]['avg_dep']:.2f}%</b> depression.
    Countries in Q4 (high spending) report <b>{group_stats.iloc[3]['avg_dep']:.2f}%</b>. Nearly identical, despite spending up to
    <b>30x more</b> on healthcare. The reported rates are flat because the system that detects depression is uneven,
    not because the disease distribution is uniform.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CHART 4 — Scatter with both extremes labeled
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">04 Two Worlds, Same Number</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-story">
Plotting every country individually shows the wide spread within each spending group.
What looks like noise is actually a story: similar reported rates emerging from completely different
healthcare realities. The countries on the left could not detect their true depression burden if they
wanted to. The countries on the right are showing what real measurement looks like.
</div>""", unsafe_allow_html=True)

if len(combined) > 5:
    coeffs = np.polyfit(np.log10(combined["HealthSpending"]), combined["Depression (%)"], 1)
    trend_x = np.logspace(np.log10(combined["HealthSpending"].min()),
                          np.log10(combined["HealthSpending"].max()), 100)
    trend_y = np.polyval(coeffs, np.log10(trend_x))
    correlation = combined["HealthSpending"].corr(combined["Depression (%)"])

    # Highlight extremes
    top3_high = combined.nlargest(3, "HealthSpending")["Entity"].tolist()
    top3_low  = combined.nsmallest(3, "HealthSpending")["Entity"].tolist()
    highlighted = top3_high + top3_low

    fig_sc = go.Figure()

    bg = combined[~combined["Entity"].isin(highlighted)]
    fig_sc.add_trace(go.Scatter(
        x=bg["HealthSpending"], y=bg["Depression (%)"],
        mode="markers",
        marker=dict(color=GRAY, size=8, opacity=0.5),
        text=bg["Entity"],
        hovertemplate="<b>%{text}</b><br>Spending: $%{x:,.0f}<br>Depression: %{y:.2f}%<extra></extra>",
        showlegend=False
    ))

    hi_low = combined[combined["Entity"].isin(top3_low)]
    fig_sc.add_trace(go.Scatter(
        x=hi_low["HealthSpending"], y=hi_low["Depression (%)"],
        mode="markers+text",
        marker=dict(color=ACCENT, size=12, opacity=0.9, line=dict(color="#005f73", width=1)),
        text=hi_low["Entity"],
        textposition="top center",
        textfont=dict(size=9, color=ACCENT),
        hovertemplate="<b>%{text}</b><br>Spending: $%{x:,.0f}<br>Depression: %{y:.2f}%<extra></extra>",
        name="Lowest spending"
    ))

    hi_high = combined[combined["Entity"].isin(top3_high)]
    fig_sc.add_trace(go.Scatter(
        x=hi_high["HealthSpending"], y=hi_high["Depression (%)"],
        mode="markers+text",
        marker=dict(color=GREEN, size=12, opacity=0.9, line=dict(color="#1a5c3a", width=1)),
        text=hi_high["Entity"],
        textposition="top center",
        textfont=dict(size=9, color=GREEN),
        hovertemplate="<b>%{text}</b><br>Spending: $%{x:,.0f}<br>Depression: %{y:.2f}%<extra></extra>",
        name="Highest spending"
    ))

    fig_sc.add_trace(go.Scatter(
        x=trend_x, y=trend_y, mode="lines",
        line=dict(color="#adb5bd", width=2, dash="dash"),
        name=f"Trend (r = {correlation:.2f})",
        hoverinfo="skip"
    ))

    fig_sc.update_layout(
        title=dict(text="Depression Rate vs Healthcare Spending Per Capita",
                   font=dict(size=12, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Inter", color=TICK),
        xaxis=dict(title="Healthcare Spending Per Capita (USD PPP, log scale)",
                   type="log", showgrid=True, gridcolor=GRID, color=TICK),
        yaxis=dict(title="Reported Depression (%)", showgrid=True, gridcolor=GRID, color=TICK,
                   tickformat=".2f", ticksuffix="%"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TICK), x=0.6, y=1.12, orientation="h"),
        height=500, margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ════════════════════════════════════════════════════════════
# Country deep dive
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">05 Explore Any Country</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-story">
Pick a country to see its depression trajectory and healthcare investment over time.
</div>""", unsafe_allow_html=True)

countries = sorted(countries_df["Entity"].unique().tolist())
default_idx = countries.index("United States") if "United States" in countries else 0
sel_country = st.selectbox("Select a country", countries, index=default_idx)

cd = countries_df[countries_df["Entity"] == sel_country].sort_values("Year")
global_avg = countries_df.groupby("Year")["Depression (%)"].mean().reset_index()

country_code = cd["Code"].iloc[0] if len(cd) > 0 else None
country_health_latest = health_df[(health_df["Code"] == country_code) &
                                    (health_df["Year"] == health_df["Year"].max())]["HealthSpending"].values
country_health_val = country_health_latest[0] if len(country_health_latest) > 0 else None

m1, m2, m3 = st.columns(3)
with m1:
    latest_rate = cd[cd["Year"] == cd["Year"].max()]["Depression (%)"].values[0] if len(cd) > 0 else 0
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{sel_country} Reported Depression</div><div class="kpi-value">{latest_rate:.2f}%</div><div class="kpi-sub">Most recent year</div></div>', unsafe_allow_html=True)
with m2:
    if country_health_val:
        st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Health Spending Per Capita</div><div class="kpi-value">${country_health_val:,.0f}</div><div class="kpi-sub">USD PPP</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Health Spending</div><div class="kpi-value">N/A</div><div class="kpi-sub">Not available</div></div>', unsafe_allow_html=True)
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
    title=dict(text=f"{sel_country} Reported Depression Rate vs Global Average",
               font=dict(size=12, color=TEXT)),
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="Inter", color=TICK),
    xaxis=dict(showgrid=False, color=TICK),
    yaxis=dict(showgrid=True, gridcolor=GRID, color=TICK,
               tickformat=".2f", ticksuffix="%"),
    legend=dict(orientation="h", y=1.12, font=dict(color=TICK)),
    height=380, margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig5, use_container_width=True)

# Conclusion
st.markdown('<div class="section-title">What This Data Reveals</div>', unsafe_allow_html=True)
st.markdown(f"""<div style="background:#f8f9fa; border-left:3px solid #0096c7; padding:18px 22px; border-radius:0 8px 8px 0; margin:14px 0; font-size:0.88rem; color:#495057; line-height:1.75;">
Two maps, one paradox. Countries that report high depression rates are not sicker than those that report low rates.
They are simply more visible. The healthcare systems that count their patients also count their suffering.
The countries with the lowest reported rates may carry the heaviest hidden burden. Mental health is universal.
What differs is whether a country has built the infrastructure to see it. Closing the gap starts with investing
in the systems that make the invisible visible.
</div>
""", unsafe_allow_html=True)
