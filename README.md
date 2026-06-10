# InfraESG Analytics

**ESG Risk Assessment & Decarbonisation Tracking for Infrastructure Investments**

An end-to-end ESG analytics platform built for infrastructure investment analysis — covering ESG scoring, SFDR PAI regulatory reporting, EU Taxonomy alignment, decarbonisation trajectory tracking, and ML-driven volatility prediction across 25 infrastructure companies spanning renewable energy, utilities, oil & gas, digital infrastructure, and social infrastructure.

## Why This Matters

Infrastructure investment is central to the energy transition and sustainable development. Institutional investors need robust ESG analytics to:

- Evaluate ESG risk during **due diligence**
- Monitor **decarbonisation trajectories** against Paris Agreement targets
- Produce **SFDR-compliant PAI disclosures** for regulators and LPs
- Quantify the relationship between **ESG performance and financial risk**

This project demonstrates these capabilities with real-world-calibrated data and production-ready tooling.

## Features

| Capability | Description |
|---|---|
| **ESG Portfolio Scoring** | E/S/G sub-scores, ESG profiles (Leader → Laggard) via K-means clustering, sector benchmarking |
| **SFDR PAI Dashboard** | All 14 mandatory Principal Adverse Impact indicators with traffic-light status and honest data-gap disclosure |
| **EU Taxonomy Alignment** | Portfolio-level taxonomy eligibility and alignment rates by sector |
| **Decarbonisation Tracker** | Scope 1+2 emissions trajectories (2020–2024), SBTi target coverage, net-zero timelines |
| **ML Risk Model** | Random Forest / Gradient Boosting volatility prediction from ESG features, leakage-free time-series cross-validation |
| **Due Diligence Scorecard** | Per-company ESG assessment with risk flags and investment readiness scoring |
| **Excel Report Export** | Multi-sheet formatted `.xlsx` workbook for investor/LP reporting |
| **Interactive Dashboard** | 6-tab Streamlit app with Plotly visualisations |

## Universe

25 infrastructure companies across 6 sectors:

- **Renewable Energy** (8): NextEra Energy, First Solar, Enphase, Brookfield Renewable, Orsted, Vestas, SolarEdge, Plug Power
- **Utilities** (6): Duke Energy, Southern Company, Dominion Energy, National Grid, Exelon, AES
- **Oil & Gas** (4): TotalEnergies, Shell, BP, Equinor
- **Infrastructure** (2): Brookfield Infrastructure, Waste Management
- **Digital Infrastructure** (4): Equinix, Digital Realty, American Tower, Crown Castle
- **Social Infrastructure** (1): Welltower

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Download stock prices and build panel dataset
python data_collection.py

# Generate Excel report
python export_excel.py

# Launch interactive dashboard
streamlit run app.py
```

The dashboard works in two modes:

- **Basic mode** (always available): ESG scoring, SFDR PAI, decarbonisation, due diligence
- **Enhanced mode** (after `data_collection.py`): + ML volatility model, financial correlations

## Project Structure

```
├── app.py                 # Streamlit dashboard (6 tabs)
├── config.py              # ESG universe data (25 companies, 15+ attributes each)
├── data_collection.py     # Stock price download & panel dataset assembly
├── analysis.py            # ML modelling, correlations, clustering
├── sfdr_pai.py            # SFDR PAI indicators & EU Taxonomy computation
├── export_excel.py        # Formatted Excel report generation
├── requirements.txt       # Python dependencies
└── data/
    ├── panel_data.csv     # Company × month panel dataset (generated)
    ├── esg_universe.csv   # ESG attributes (generated)
    ├── stock_prices.csv   # Daily prices (generated)
    ├── InfraESG_Report.xlsx  # Excel report (generated)
    └── charts/            # Static chart exports
```

## Methodology

**Data:** Company-level ESG attributes calibrated against publicly reported sustainability metrics (CDP, company ESG reports, proxy statements; FY2023–FY2024 representative values). Stock prices from Yahoo Finance. For production deployment, integrate with MSCI ESG, Sustainalytics, or Bloomberg ESG data feeds.

**Financed emissions (PAI 1–2):** Standard attribution approach — each company's emissions are attributed in proportion to the portfolio's ownership share (investment / company value), computed for an illustrative €1bn portfolio weighted by market cap. Indicators where issuer-level data is not collected (PAI 7–9) are disclosed as data gaps rather than silently zeroed — mirroring how real reporting teams handle first-year coverage under SFDR's best-efforts provisions.

**ML Pipeline:**

- Features: ESG score, E/S/G sub-scores, carbon intensity, renewable energy %, taxonomy alignment, board diversity, board independence
- Target: Annualised realised volatility (monthly, √252 scaled)
- Models: Linear Regression, Random Forest (200 trees), Gradient Boosting
- Validation: TimeSeriesSplit 5-fold cross-validation on a chronologically sorted panel — the model always trains on the past and tests on the future, so reported CV-RMSE is a genuine out-of-sample metric
- ~1,500+ company-month observations (25 companies × 60+ months)

**Regulatory Framework (as of 2026):**

- **SFDR** (EU 2019/2088, Delegated Regulation 2022/1288) — the 14 mandatory PAI indicators implemented here remain the applicable disclosure standard. The Commission's **SFDR 2.0 proposal** (November 2025) would streamline product-level PAIs and align them with CSRD data; this project's indicator engine is structured so thresholds and indicator sets can be swapped without touching the rest of the pipeline.
- **EU Taxonomy** (EU 2020/852) — 6 environmental objectives; eligibility vs alignment tracked per company.
- **Omnibus / CSRD** — the 2026 Omnibus package narrowed CSRD scope (1,000+ employees, €450m+ turnover) and simplified ESRS datapoints; most companies in this universe remain in scope as large issuers.
- **SBTi** — Science Based Targets initiative validation tracking.

## Tech Stack

Python, Pandas, Scikit-learn, Plotly, Streamlit, yfinance, openpyxl

## Author

**Mridul Daga**

- [LinkedIn](https://www.linkedin.com/in/mriduldaga)
- Email: mridul.daga@edhec.com
