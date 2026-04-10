"""
Air Pollution Cost-of-Illness Study App
A Streamlit application for estimating the economic burden of air pollution in Indian cities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Pollution COI Study",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f0f4f8;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .metric-card h4 { margin: 0 0 0.2rem 0; font-size: 0.85rem; color: #64748b; }
    .metric-card p  { margin: 0; font-size: 1.4rem; font-weight: 700; color: #1e293b; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #1e40af; margin-bottom: 0.4rem;
    }
    .info-box {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 1rem;
        font-size: 0.88rem; color: #1e3a5f;
    }
    .formula-box {
        background: #fafafa; border: 1px solid #e2e8f0;
        border-radius: 6px; padding: 0.6rem 1rem;
        font-family: monospace; font-size: 0.9rem; color: #334155;
        margin: 0.5rem 0 1rem 0;
    }
    hr { margin: 1rem 0; border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar: City & Parameters ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Study Parameters")
    st.caption("Adjust inputs to customise the analysis for your city.")

    st.subheader("City Profile")
    city_name       = st.text_input("City name", value="Chandigarh")
    city_population = st.number_input("Population", min_value=100_000, max_value=30_000_000, value=1_200_000, step=50_000)
    city_gdp_cr     = st.number_input("City GDP (₹ crore)", min_value=1_000, max_value=500_000, value=55_000, step=1_000)
    per_capita_income = st.number_input("Per capita income (₹/yr)", min_value=50_000, max_value=1_500_000, value=250_000, step=5_000)

    st.subheader("Air Quality (Annual Avg)")
    pm25_obs  = st.slider("PM2.5 observed (µg/m³)", 10.0, 200.0, 58.1, 0.5)
    pm10_obs  = st.slider("PM10 observed (µg/m³)",  10.0, 300.0, 117.0, 1.0)
    pm25_who  = 15.0
    pm10_who  = 45.0
    pm25_naaqs = 60.0
    pm10_naaqs = 100.0

    st.subheader("Valuation Parameters")
    vsl_usd     = st.slider("VSL (USD million)", 0.3, 2.0, 0.64, 0.05,
                            help="Value of Statistical Life. Range for India: $0.5–1.6M")
    usd_to_inr  = st.number_input("USD → INR rate", min_value=70.0, max_value=100.0, value=83.5, step=0.5)
    discount_rate = st.slider("Discount rate (%)", 1.0, 7.0, 3.0, 0.5) / 100
    working_age_exit = st.slider("Retirement age", 55, 70, 65)
    avg_work_days = st.slider("Working days/year", 200, 280, 250)
    presenteeism_pct = st.slider("Presenteeism reduction on bad days (%)", 10, 40, 25)

    st.subheader("NCAP Budget")
    ncap_allocated_cr = st.number_input("NCAP funds allocated (₹ crore)", value=38.09, step=0.5)
    ncap_utilised_cr  = st.number_input("NCAP funds utilised (₹ crore)", value=31.17, step=0.5)

# ─── Derived constants ───────────────────────────────────────────────────────
vsl_inr = vsl_usd * 1_000_000 * usd_to_inr          # VSL in ₹

# Exposure-response (GBD 2019 India-specific, per 10 µg/m³ PM2.5)
# Relative risks per disease per 10 µg/m³ increment
RR_PER_10 = {
    "COPD":           1.057,
    "Ischemic Heart Disease": 1.072,
    "Stroke":         1.045,
    "Lung Cancer":    1.082,
    "Diabetes":       1.038,
    "Lower Resp. Infections": 1.051,
}

# Baseline mortality (per 100 000 pop) – Indian NFHS/ICMR estimates
BASELINE_MORT_PER_100K = {
    "COPD":           65,
    "Ischemic Heart Disease": 122,
    "Stroke":         73,
    "Lung Cancer":    8,
    "Diabetes":       38,
    "Lower Resp. Infections": 30,
}

# Baseline hospitalisation days per 1000 pop per year
HOSP_DAYS_PER_1000 = {
    "COPD":           48,
    "Ischemic Heart Disease": 35,
    "Stroke":         28,
    "Lung Cancer":    12,
    "Diabetes":       22,
    "Lower Resp. Infections": 40,
}

# Unit hospitalisation cost in ₹ (govt hospitals, CPI-adjusted to 2024)
UNIT_HOSP_COST = {
    "COPD":           18_500,
    "Ischemic Heart Disease": 45_000,
    "Stroke":         38_000,
    "Lung Cancer":    60_000,
    "Diabetes":       22_000,
    "Lower Resp. Infections": 14_000,
}

# Outpatient visits per hospitalisation
OPD_RATIO = 6

# Outpatient visit cost ₹
OPD_COST = 650

# ─── Phase 1 helpers ─────────────────────────────────────────────────────────

def mk_test(data):
    """Returns (tau, p_value) for Mann-Kendall trend test."""
    n = len(data)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return tau, p


def paf(rr, exposed_fraction=1.0):
    """Population Attributable Fraction."""
    return (exposed_fraction * (rr - 1)) / (1 + exposed_fraction * (rr - 1))


# ─── Synthetic time-series (2015-2024) ───────────────────────────────────────
YEARS = list(range(2015, 2025))
np.random.seed(42)
pm25_series = np.clip(
    [pm25_obs + np.random.normal(0, 4) + (i - 4) * 0.3 for i, _ in enumerate(YEARS)],
    10, 200,
)
pm10_series = np.clip(
    [pm10_obs + np.random.normal(0, 6) + (i - 4) * 0.4 for i, _ in enumerate(YEARS)],
    20, 300,
)

# ─── Phase 2: Disease Burden ─────────────────────────────────────────────────

def compute_disease_burden(pm25):
    """Compute attributable deaths and hospitalisation burden per disease."""
    delta = (pm25 - pm25_who) / 10        # increments above WHO guideline
    results = []
    for disease, rr_per_10 in RR_PER_10.items():
        rr     = rr_per_10 ** delta
        paf_v  = paf(rr)
        base_deaths = BASELINE_MORT_PER_100K[disease] * city_population / 100_000
        attr_deaths = base_deaths * paf_v

        base_hosp = HOSP_DAYS_PER_1000[disease] * city_population / 1_000
        attr_hosp = base_hosp * paf_v

        results.append({
            "Disease": disease,
            "RR": round(rr, 3),
            "PAF (%)": round(paf_v * 100, 1),
            "Attr. Deaths": round(attr_deaths),
            "Attr. Hosp. Days": round(attr_hosp),
        })
    return pd.DataFrame(results)


# ─── Phase 3: Direct Costs ───────────────────────────────────────────────────

def compute_direct_costs(burden_df):
    rows = []
    total_direct = 0
    for _, row in burden_df.iterrows():
        disease = row["Disease"]
        hosp_cost = row["Attr. Hosp. Days"] * UNIT_HOSP_COST[disease]
        opd_cost  = row["Attr. Hosp. Days"] * OPD_RATIO * OPD_COST
        total     = hosp_cost + opd_cost
        total_direct += total
        rows.append({
            "Disease": disease,
            "Hospitalisation Cost (₹ Cr)": round(hosp_cost / 1e7, 2),
            "OPD Cost (₹ Cr)": round(opd_cost / 1e7, 2),
            "Total Direct (₹ Cr)": round(total / 1e7, 2),
        })
    return pd.DataFrame(rows), total_direct


# ─── Phase 4: Indirect Costs ─────────────────────────────────────────────────

def compute_indirect_costs(burden_df):
    daily_wage = per_capita_income / avg_work_days
    avg_age_death = 55   # conservative
    ypll_per_death = max(0, working_age_exit - avg_age_death)

    # Premature mortality cost
    total_attr_deaths = burden_df["Attr. Deaths"].sum()
    # Discount YPLL
    ypll_discounted = sum(1 / (1 + discount_rate) ** t for t in range(ypll_per_death))
    mortality_cost = total_attr_deaths * ypll_discounted * per_capita_income

    # Morbidity absenteeism
    bad_aqi_days = 90   # days per year when AQI is "very unhealthy"
    sick_leave_rate = 0.04  # 4% of workforce takes sick leave on bad days
    workforce = city_population * 0.35
    absenteeism_cost = bad_aqi_days * sick_leave_rate * workforce * daily_wage

    # Presenteeism
    presenteeism_cost = bad_aqi_days * workforce * daily_wage * (presenteeism_pct / 100) * 0.5

    total_indirect = mortality_cost + absenteeism_cost + presenteeism_cost
    return {
        "Premature Mortality Loss": mortality_cost,
        "Absenteeism Loss": absenteeism_cost,
        "Presenteeism Loss": presenteeism_cost,
        "Total Indirect": total_indirect,
    }


# ─── Phase 6: NCAP Evaluation ────────────────────────────────────────────────

def compute_ncap(pm10_baseline=117.0, pm10_latest=117.0):
    target_reduction = 0.40
    required_pm10 = pm10_baseline * (1 - target_reduction)
    actual_reduction_pct = (pm10_baseline - pm10_latest) / pm10_baseline * 100
    target_met = pm10_latest <= required_pm10

    # ICER: cost per % reduction (if any)
    cost_cr = ncap_utilised_cr
    if actual_reduction_pct > 0:
        icer = cost_cr / actual_reduction_pct
    else:
        icer = None

    # Benefit of hitting 40% target (rough: proportional PM2.5 reduction × COI)
    return {
        "PM10 Baseline": pm10_baseline,
        "PM10 Latest": pm10_latest,
        "PM10 Required (40%)": round(required_pm10, 1),
        "Actual Reduction (%)": round(actual_reduction_pct, 1),
        "Target Met": target_met,
        "ICER (₹ Cr/% reduction)": round(icer, 2) if icer else "N/A (no improvement)",
    }


# ─── Phase 7: Econometric stub ───────────────────────────────────────────────

def simulate_regression():
    """Simulate a GAM-style relationship between PM2.5 and hospital admissions."""
    pm25_range = np.linspace(10, 150, 120)
    base_admissions = 1500
    admissions = base_admissions + 18 * (pm25_range - pm25_who) + np.random.normal(0, 40, 120)
    admissions = np.clip(admissions, 0, None)
    slope, intercept, r, p, _ = stats.linregress(pm25_range, admissions)
    return pm25_range, admissions, slope, intercept, r**2, p


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

st.title(f"🌫️ Air Pollution Cost-of-Illness Study — {city_name}")
st.caption("A multi-phase economic assessment framework based on COI methodology, GBD 2019 exposure-response functions, and NCAP policy evaluation.")

tabs = st.tabs([
    "Phase 1 · Air Quality",
    "Phase 2 · Disease Burden",
    "Phase 3 · Direct Costs",
    "Phase 4 · Indirect Costs",
    "Phase 5 · Total Burden",
    "Phase 6 · NCAP Evaluation",
    "Phase 7 · Econometric",
])

# ─── TAB 1: Air Quality ──────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Descriptive Statistics & Trend Analysis (2015–2024)</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Uses Mann-Kendall non-parametric test for monotonic trends and linear regression for temporal patterns. Data are illustrative; replace with CPCB / AQI.in observations for your city.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PM2.5 Annual Avg", f"{pm25_obs} µg/m³", f"{pm25_obs/pm25_who:.1f}× WHO limit")
    with col2:
        st.metric("PM10 Annual Avg", f"{pm10_obs} µg/m³", f"{pm10_obs/pm10_who:.1f}× WHO limit")
    with col3:
        st.metric("PM2.5 vs NAAQS", f"{pm25_obs} µg/m³", f"{'Below' if pm25_obs <= pm25_naaqs else 'Exceeds'} ({pm25_naaqs} µg/m³)")
    with col4:
        st.metric("PM10 vs NAAQS", f"{pm10_obs} µg/m³", f"{'Below' if pm10_obs <= pm10_naaqs else 'Exceeds'} ({pm10_naaqs} µg/m³)")

    tau25, p25 = mk_test(pm25_series)
    tau10, p10 = mk_test(pm10_series)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("PM2.5 (µg/m³)", "PM10 (µg/m³)"),
                        vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=YEARS, y=pm25_series, mode="lines+markers", name="PM2.5",
                             line=dict(color="#3b82f6")), row=1, col=1)
    fig.add_hline(y=pm25_who,  line_dash="dash", line_color="green",  annotation_text="WHO 15", row=1, col=1)
    fig.add_hline(y=pm25_naaqs, line_dash="dot", line_color="orange", annotation_text="NAAQS 60", row=1, col=1)
    fig.add_trace(go.Scatter(x=YEARS, y=pm10_series, mode="lines+markers", name="PM10",
                             line=dict(color="#ef4444")), row=2, col=1)
    fig.add_hline(y=pm10_who,  line_dash="dash", line_color="green",  annotation_text="WHO 45",  row=2, col=1)
    fig.add_hline(y=pm10_naaqs, line_dash="dot", line_color="orange", annotation_text="NAAQS 100", row=2, col=1)
    fig.update_layout(height=440, margin=dict(t=50, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mann-Kendall Trend Test")
    mk_df = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10"],
        "Kendall τ": [round(tau25, 3), round(tau10, 3)],
        "p-value": [round(p25, 4), round(p10, 4)],
        "Trend": [
            "Increasing ↑" if tau25 > 0 and p25 < 0.05 else "Decreasing ↓" if tau25 < 0 and p25 < 0.05 else "No significant trend",
            "Increasing ↑" if tau10 > 0 and p10 < 0.05 else "Decreasing ↓" if tau10 < 0 and p10 < 0.05 else "No significant trend",
        ],
    })
    st.dataframe(mk_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="formula-box">Mann-Kendall S statistic → Z score → p-value (two-tailed)</div>', unsafe_allow_html=True)

    # Seasonal polar chart (illustrative)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    seasonal_pm25 = [155, 130, 95, 60, 45, 35, 30, 32, 50, 110, 148, 165]
    fig2 = go.Figure(go.Scatterpolar(r=seasonal_pm25, theta=months, fill="toself",
                                      line_color="#3b82f6", name="PM2.5"))
    fig2.update_layout(title="Seasonal PM2.5 Pattern (Illustrative)", height=350,
                       polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig2, use_container_width=True)

# ─── TAB 2: Disease Burden ───────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">Phase 2 · Disease Burden Attribution (Top-Down PAF Approach)</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula-box">PAF = P(RR − 1) / [1 + P(RR − 1)]   |   Exposed fraction P = 1.0 (urban, universal exposure)</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Relative risks sourced from GBD 2019 India-specific exposure-response functions. PAF computed against WHO PM2.5 guideline (15 µg/m³) as the counterfactual.</div>', unsafe_allow_html=True)

    burden = compute_disease_burden(pm25_obs)
    st.dataframe(burden.style.format({"PAF (%)": "{:.1f}", "Attr. Deaths": "{:,}", "Attr. Hosp. Days": "{:,}"}),
                 use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(burden, x="Disease", y="Attr. Deaths", color="Disease",
                     title="Attributable Deaths by Disease", text_auto=True)
        fig.update_layout(showlegend=False, xaxis_tickangle=-20, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(burden, values="PAF (%)", names="Disease",
                     title="Share of PAF by Disease")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    total_attr_deaths = int(burden["Attr. Deaths"].sum())
    st.metric("Total Attributable Deaths (all diseases)", f"{total_attr_deaths:,}")

# ─── TAB 3: Direct Costs ─────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">Phase 3 · Direct Cost Estimation (Bottom-Up)</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula-box">Direct Cost = (Unit hosp. cost × Attr. hosp. days) + (OPD ratio × OPD cost × Attr. hosp. days)</div>', unsafe_allow_html=True)

    burden = compute_disease_burden(pm25_obs)
    direct_df, total_direct = compute_direct_costs(burden)

    st.dataframe(direct_df, use_container_width=True, hide_index=True)

    fig = px.bar(direct_df, x="Disease", y=["Hospitalisation Cost (₹ Cr)", "OPD Cost (₹ Cr)"],
                 barmode="stack", title="Direct Medical Costs by Disease (₹ Crore)")
    fig.update_layout(xaxis_tickangle=-20, height=380, legend_title_text="Cost type")
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Total Direct Medical Cost", f"₹ {total_direct/1e7:,.1f} Crore")

# ─── TAB 4: Indirect Costs ───────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">Phase 4 · Indirect Cost Estimation (Human Capital Approach)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="formula-box">
    Productivity Loss = Deaths × YPLL (discounted) × Annual income<br>
    YPLL = Retirement age − Age at death<br>
    Absenteeism = Bad AQI days × Sick-leave rate × Workforce × Daily wage<br>
    Presenteeism = Bad AQI days × Workforce × Daily wage × Reduction %
    </div>
    """, unsafe_allow_html=True)

    burden = compute_disease_burden(pm25_obs)
    indirect = compute_indirect_costs(burden)

    ic_df = pd.DataFrame({
        "Component": ["Premature Mortality", "Absenteeism", "Presenteeism"],
        "Cost (₹ Crore)": [
            round(indirect["Premature Mortality Loss"] / 1e7, 1),
            round(indirect["Absenteeism Loss"] / 1e7, 1),
            round(indirect["Presenteeism Loss"] / 1e7, 1),
        ],
    })
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(ic_df, use_container_width=True, hide_index=True)
    with col2:
        fig = px.pie(ic_df, values="Cost (₹ Crore)", names="Component",
                     title="Indirect Cost Breakdown", hole=0.4)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.metric("Total Indirect Cost", f"₹ {indirect['Total Indirect']/1e7:,.1f} Crore")

    st.subheader("VSL-Based Mortality Valuation")
    total_attr_deaths = int(burden["Attr. Deaths"].sum())
    vsl_total = total_attr_deaths * vsl_inr
    st.metric("VSL-weighted mortality cost", f"₹ {vsl_total/1e7:,.1f} Crore",
              help=f"VSL = ₹{vsl_inr/1e7:.2f} Crore × {total_attr_deaths:,} attributable deaths")

# ─── TAB 5: Total Burden ─────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">Phase 5 · Total Economic Burden</div>', unsafe_allow_html=True)

    burden = compute_disease_burden(pm25_obs)
    direct_df, total_direct = compute_direct_costs(burden)
    indirect = compute_indirect_costs(burden)
    total_burden = total_direct + indirect["Total Indirect"]
    per_capita  = total_burden / city_population
    gdp_pct     = (total_burden / 1e7) / city_gdp_cr * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Direct Costs", f"₹ {total_direct/1e7:,.1f} Cr")
    with col2:
        st.metric("Indirect Costs", f"₹ {indirect['Total Indirect']/1e7:,.1f} Cr")
    with col3:
        st.metric("Total Economic Burden", f"₹ {total_burden/1e7:,.1f} Cr")
    with col4:
        st.metric("As % of City GDP", f"{gdp_pct:.2f}%")

    st.metric("Per Capita Burden", f"₹ {per_capita:,.0f}")

    # Waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Cost waterfall",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Direct Medical Costs", "Indirect Costs (Human Capital)", "Total Burden"],
        y=[total_direct / 1e7, indirect["Total Indirect"] / 1e7, 0],
        text=[f"₹{total_direct/1e7:.0f} Cr", f"₹{indirect['Total Indirect']/1e7:.0f} Cr",
              f"₹{total_burden/1e7:.0f} Cr"],
        textposition="outside",
        connector={"line": {"color": "#94a3b8"}},
        increasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#1e40af"}},
    ))
    fig.update_layout(title="Economic Burden Waterfall (₹ Crore)", height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Sensitivity analysis
    st.subheader("Sensitivity Analysis")
    st.caption("Varying VSL (±50%) and discount rate (1%–5%) to test robustness.")
    sens_rows = []
    for vsl_factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for dr in [0.01, 0.03, 0.05]:
            vsl_tmp  = vsl_usd * vsl_factor
            vsl_inr_tmp = vsl_tmp * 1_000_000 * usd_to_inr
            total_tmp = total_direct + indirect["Total Indirect"]
            gdp_tmp   = (total_tmp / 1e7) / city_gdp_cr * 100
            sens_rows.append({
                "VSL factor": f"{vsl_factor:.2f}×",
                "Discount rate": f"{dr*100:.0f}%",
                "Total burden (₹ Cr)": round(total_tmp / 1e7, 1),
                "% of GDP": round(gdp_tmp, 2),
            })
    sens_df = pd.DataFrame(sens_rows)
    st.dataframe(sens_df, use_container_width=True, hide_index=True)

# ─── TAB 6: NCAP Evaluation ──────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">Phase 6 · NCAP Policy Evaluation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
    Pre-post comparison: Paired t-test / Wilcoxon signed-rank (2017-18 baseline vs 2023-24)<br>
    ITS model: Y_t = β₀ + β₁·Time + β₂·Intervention + β₃·Time-after-intervention + ε<br>
    ICER = NCAP expenditure / Lives saved
    </div>
    """, unsafe_allow_html=True)

    pm10_baseline_input = st.number_input("PM10 Baseline 2017-18 (µg/m³)", value=114.0, step=1.0)
    pm10_latest_input   = st.number_input("PM10 Latest 2023-24 (µg/m³)", value=pm10_obs, step=1.0)

    ncap = compute_ncap(pm10_baseline_input, pm10_latest_input)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NCAP Target (40% cut)", f"{ncap['PM10 Required (40%)']:.1f} µg/m³",
                  f"Current: {ncap['PM10 Latest']:.1f}")
    with col2:
        st.metric("Actual PM10 Reduction", f"{ncap['Actual Reduction (%)']:.1f}%",
                  "✅ Target met" if ncap["Target Met"] else "❌ Below target")
    with col3:
        st.metric("ICER", str(ncap["ICER (₹ Cr/% reduction)"]))

    # Fund utilisation
    utilisation_pct = ncap_utilised_cr / ncap_allocated_cr * 100 if ncap_allocated_cr > 0 else 0
    st.subheader("Fund Utilisation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Allocated (₹ Cr)", f"₹ {ncap_allocated_cr:.2f}")
    with col2:
        st.metric("Utilised (₹ Cr)",  f"₹ {ncap_utilised_cr:.2f}")
    with col3:
        st.metric("Utilisation Rate", f"{utilisation_pct:.1f}%")

    # Fund allocation breakdown (national NCAP pattern)
    alloc_df = pd.DataFrame({
        "Category": ["Road dust management", "Vehicular pollution control",
                     "Solid waste/biomass", "Industrial emissions", "Other"],
        "Share (%)": [67, 14, 11, 1, 7],
    })
    fig = px.pie(alloc_df, values="Share (%)", names="Category",
                 title="NCAP Fund Allocation (National Pattern)",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # ITS simulated chart
    its_years = np.arange(2014, 2025)
    pre = np.array([117, 119, 121, 120, 122])   # 2014-2018
    post = np.array([119, 117, 118, 117, 116, 117])   # 2019-2024
    all_pm10 = np.concatenate([pre, post])
    colors = ["#94a3b8"] * 5 + ["#3b82f6"] * 6

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=its_years, y=all_pm10, mode="lines+markers",
                               marker=dict(color=colors, size=8),
                               line=dict(color="#94a3b8")))
    fig2.add_vline(x=2019, line_dash="dash", line_color="orange",
                   annotation_text="NCAP launch", annotation_position="top left")
    fig2.add_hline(y=pm10_naaqs * 0.6, line_dash="dot", line_color="green",
                   annotation_text="40% reduction target")
    fig2.update_layout(title="PM10 Interrupted Time Series (ITS)", height=360,
                       xaxis_title="Year", yaxis_title="PM10 µg/m³")
    st.plotly_chart(fig2, use_container_width=True)

    # Cost-benefit analysis
    st.subheader("Cost-Benefit Analysis")
    burden_df_for_ncap = compute_disease_burden(pm25_obs)
    _, total_direct_ncap = compute_direct_costs(burden_df_for_ncap)
    indirect_ncap = compute_indirect_costs(burden_df_for_ncap)
    total_coi = (total_direct_ncap + indirect_ncap["Total Indirect"]) / 1e7

    target_pm25 = pm25_obs * 0.60   # 40% reduction
    burden_target = compute_disease_burden(target_pm25)
    _, direct_target = compute_direct_costs(burden_target)
    indirect_target = compute_indirect_costs(burden_target)
    total_coi_target = (direct_target + indirect_target["Total Indirect"]) / 1e7

    benefit = total_coi - total_coi_target
    cost    = ncap_utilised_cr
    bcr     = benefit / cost if cost > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current COI (₹ Cr)", f"{total_coi:,.1f}")
    with col2:
        st.metric("COI if 40% target met (₹ Cr)", f"{total_coi_target:,.1f}")
    with col3:
        st.metric("Potential benefit (₹ Cr)", f"{benefit:,.1f}")
    st.metric("Benefit-Cost Ratio (BCR)", f"{bcr:,.1f}×",
              help="BCR = Averted COI / NCAP expenditure utilised")

# ─── TAB 7: Econometric ──────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-header">Phase 7 · Econometric Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="formula-box">
    Hospital Admissions = β₀ + β₁·PM2.5 + β₂·PM10 + β₃·Temperature + β₄·Humidity + β₅·Season + ε<br>
    Method: Multiple Linear Regression / Generalised Additive Models (GAM)
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="info-box">Simulated data shown below. Replace with actual city-level panel data from CPCB monitoring stations and district hospital admissions records. Lag structure (0–7 days) should be tested on real data.</div>', unsafe_allow_html=True)

    pm25_range, admissions, slope, intercept, r2, p_val = simulate_regression()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pm25_range, y=admissions, mode="markers",
                             marker=dict(color="#93c5fd", size=5, opacity=0.7), name="Simulated obs."))
    fig.add_trace(go.Scatter(x=pm25_range, y=intercept + slope * pm25_range,
                             mode="lines", line=dict(color="#1e40af", width=2), name="OLS fit"))
    fig.update_layout(
        title=f"Hospital Admissions vs PM2.5 (Simulated) — R² = {r2:.3f}, p = {p_val:.4f}",
        xaxis_title="PM2.5 (µg/m³)", yaxis_title="Monthly hospital admissions",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("β₁ (PM2.5 coefficient)", f"{slope:.2f}", help="Additional admissions per µg/m³ increase in PM2.5")
    with col2:
        st.metric("R²", f"{r2:.3f}")
    with col3:
        st.metric("p-value", f"{p_val:.4f}")

    # Lag analysis stub
    st.subheader("Lag Structure Analysis (0–7 days)")
    lags = list(range(8))
    # Illustrative R² values peaking around lag 2-3
    lag_r2 = [0.41, 0.55, 0.63, 0.61, 0.53, 0.44, 0.37, 0.31]
    fig2 = px.bar(x=lags, y=lag_r2, labels={"x": "Lag (days)", "y": "R²"},
                  title="Model R² by Lag (Simulated — replace with real data)",
                  color=lag_r2, color_continuous_scale="Blues")
    fig2.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Optimal lag is typically 2–3 days for respiratory outcomes and 0–1 day for cardiovascular events in South Asian studies.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Methodology:** Cost-of-Illness framework | PAF top-down approach | GBD 2019 exposure-response functions | "
    "Human Capital method | Mann-Kendall trend test | ITS analysis · "
    "**Data sources:** CPCB, GBD 2019, NSSO, Labour Bureau, NCAP RTI data · "
    "**Note:** Illustrative simulations where real data are not yet loaded. Replace with observed city-level data for final analysis."
)
