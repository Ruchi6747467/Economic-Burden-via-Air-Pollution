# 🌫️ Air Pollution Cost-of-Illness (COI) Study App

An interactive Streamlit application for estimating and communicating the economic burden of air pollution in Indian non-attainment cities. Built around the seven-phase methodology of the working paper *"What Economic Burden Does Air Pollution Create in a City?"*

---

## Features

| Tab | Phase | What it does |
|-----|-------|-------------|
| 📊 Air Quality | Phase 1 | Descriptive stats, Mann-Kendall trend test, seasonal polar chart for PM2.5 / PM10 |
| 🦠 Disease Burden | Phase 2 | Population Attributable Fraction (PAF) using GBD 2019 exposure-response functions for six diseases |
| 🏥 Direct Costs | Phase 3 | Bottom-up hospitalisation + OPD cost estimation per disease |
| 💼 Indirect Costs | Phase 4 | Human Capital approach — premature mortality (discounted YPLL), absenteeism, presenteeism |
| 💰 Total Burden | Phase 5 | Waterfall chart, per-capita burden, % of city GDP, sensitivity analysis across VSL × discount rate |
| 📋 NCAP Evaluation | Phase 6 | ITS model visualisation, ICER, fund utilisation, benefit-cost ratio for 40% target |
| 📈 Econometric | Phase 7 | Simulated GAM/OLS regression of PM2.5 vs hospital admissions, lag-structure analysis |

All city parameters, air quality readings, and valuation assumptions are adjustable in real time from the **sidebar**.

---

## Quickstart

### 1. Clone / download
```bash
git clone <your-repo-url>
cd air-pollution-coi
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Replacing Simulated Data with Real Observations

The app ships with **illustrative / synthetic data** so every phase renders immediately. To use your city's actual figures:

### Air Quality (Phase 1)
Replace the `pm25_series` / `pm10_series` NumPy arrays near line 100 with your CPCB / AQI.in annual averages:
```python
YEARS      = [2015, 2016, ..., 2024]
pm25_series = [52.3, 55.1, ...]   # your observed annual means
pm10_series = [110.0, 114.2, ...]
```

### Disease Burden (Phase 2)
The `BASELINE_MORT_PER_100K` and `HOSP_DAYS_PER_1000` dictionaries near line 80 use national NFHS/ICMR averages. Swap these for district-level HMIS data if available.

### Direct Costs (Phase 3)
Unit costs in `UNIT_HOSP_COST` are CPI-adjusted 2024 government-hospital estimates. Update with local hospital billing data or NSSO health expenditure survey figures.

### Indirect Costs (Phase 4)
`per_capita_income` and `avg_work_days` are set in the sidebar. `bad_aqi_days` (default 90) should reflect the number of days per year when your city's AQI exceeds the "Very Unhealthy" threshold — adjust in the `compute_indirect_costs()` function.

### NCAP Evaluation (Phase 6)
Enter your city's actual NCAP allocation and utilisation figures in the sidebar, and update the `pm10_baseline_input` / `pm10_latest_input` fields with monitored values.

### Econometric (Phase 7)
Replace the `simulate_regression()` output with a real regression run on panel data linking daily hospital admissions to lagged pollutant concentrations, temperature, humidity, and seasonal dummies.

---

## Methodology Reference

| Component | Approach | Key formula |
|-----------|----------|-------------|
| Trend detection | Mann-Kendall (non-parametric) | S-statistic → Z → p-value |
| Disease attribution | Top-down PAF | `PAF = P(RR−1) / [1 + P(RR−1)]` |
| Direct costs | Bottom-up unit costing | Hosp. days × unit cost + OPD visits × OPD cost |
| Indirect — mortality | Human Capital (discounted YPLL) | `Deaths × Σ(1/(1+r)^t) × annual income` |
| Indirect — absenteeism | Wage-day loss | `Bad-AQI days × sick-leave rate × workforce × daily wage` |
| Mortality valuation | VSL (hedonic / benefit transfer) | Majumder & Madheswaran (2018); Guttikunda (2024) |
| Policy evaluation | ITS + ICER + BCR | `ICER = expenditure / % PM10 reduction` |
| Exposure-response | OLS / GAM | `Admissions = β₀ + β₁PM2.5 + β₂PM10 + controls + ε` |

Standards used:
- **WHO 2021:** PM2.5 = 15 µg/m³, PM10 = 45 µg/m³
- **NAAQS (India):** PM2.5 = 60 µg/m³, PM10 = 100 µg/m³

---

## Data Sources (for real-data integration)

| Data | Source |
|------|--------|
| Air quality monitoring | CPCB Continuous Ambient Air Quality Monitoring (CAAQM); [AQI.in](https://www.aqi.in/in) |
| Mortality / morbidity | HMIS, NFHS-5 (2019-21), GBD 2019 India state-level estimates |
| Health expenditure | NSSO 75th Round (2017-18), National Health Accounts |
| Wages / productivity | Labour Bureau wage statistics, PLFS |
| NCAP allocations | MoEFCC RTI data; Malhotra et al. (2025) |
| VSL estimates | Majumder & Madheswaran (2018); Guttikunda (2024) |

---

## Project Structure

```
air-pollution-coi/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## License

For academic and policy research use. Please cite the underlying working paper and the GBD 2019 data when publishing results derived from this tool.
