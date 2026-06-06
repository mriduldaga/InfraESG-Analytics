"""
Infrastructure ESG Universe — Company-level ESG dataset.

Data based on publicly reported sustainability metrics from company annual/ESG
reports, CDP disclosures, and proxy statements (FY2023 representative values).
For production use, integrate with MSCI, Sustainalytics, or Bloomberg ESG.
"""

import pandas as pd
import numpy as np

# ── Company Universe ─────────────────────────────────────────────────────────

COMPANIES = [
    # ── Renewable Energy ─────────────────────────────────────────────────
    {
        "name": "NextEra Energy", "ticker": "NEE",
        "sector": "Renewable Energy", "subsector": "Integrated Renewables",
        "country": "US", "market_cap_bn": 150.2,
        "esg_score": 75, "env_score": 82, "social_score": 68, "gov_score": 75,
        "scope1_ktco2e": 24500, "scope2_ktco2e": 480, "scope3_ktco2e": 5200,
        "revenue_mn": 28100, "renewable_energy_pct": 65,
        "employees": 16200, "women_mgmt_pct": 28, "board_diversity_pct": 33,
        "board_independence_pct": 92,
        "taxonomy_eligible_pct": 72, "taxonomy_aligned_pct": 58,
        "has_sbti": True, "net_zero_year": 2045, "controversy_score": 1,
    },
    {
        "name": "First Solar", "ticker": "FSLR",
        "sector": "Renewable Energy", "subsector": "Solar Manufacturing",
        "country": "US", "market_cap_bn": 22.8,
        "esg_score": 71, "env_score": 78, "social_score": 65, "gov_score": 70,
        "scope1_ktco2e": 95, "scope2_ktco2e": 340, "scope3_ktco2e": 1800,
        "revenue_mn": 3500, "renewable_energy_pct": 40,
        "employees": 7200, "women_mgmt_pct": 25, "board_diversity_pct": 30,
        "board_independence_pct": 88,
        "taxonomy_eligible_pct": 92, "taxonomy_aligned_pct": 78,
        "has_sbti": True, "net_zero_year": 2040, "controversy_score": 1,
    },
    {
        "name": "Enphase Energy", "ticker": "ENPH",
        "sector": "Renewable Energy", "subsector": "Solar Technology",
        "country": "US", "market_cap_bn": 16.5,
        "esg_score": 66, "env_score": 72, "social_score": 60, "gov_score": 65,
        "scope1_ktco2e": 5, "scope2_ktco2e": 18, "scope3_ktco2e": 420,
        "revenue_mn": 2290, "renewable_energy_pct": 55,
        "employees": 3400, "women_mgmt_pct": 22, "board_diversity_pct": 25,
        "board_independence_pct": 85,
        "taxonomy_eligible_pct": 95, "taxonomy_aligned_pct": 82,
        "has_sbti": False, "net_zero_year": 2040, "controversy_score": 0,
    },
    {
        "name": "Brookfield Renewable", "ticker": "BEPC",
        "sector": "Renewable Energy", "subsector": "Diversified Renewables",
        "country": "CA", "market_cap_bn": 20.1,
        "esg_score": 78, "env_score": 85, "social_score": 70, "gov_score": 78,
        "scope1_ktco2e": 120, "scope2_ktco2e": 45, "scope3_ktco2e": 310,
        "revenue_mn": 4860, "renewable_energy_pct": 95,
        "employees": 3200, "women_mgmt_pct": 30, "board_diversity_pct": 36,
        "board_independence_pct": 90,
        "taxonomy_eligible_pct": 96, "taxonomy_aligned_pct": 90,
        "has_sbti": True, "net_zero_year": 2035, "controversy_score": 0,
    },
    {
        "name": "Orsted", "ticker": "DNNGY",
        "sector": "Renewable Energy", "subsector": "Offshore Wind",
        "country": "DK", "market_cap_bn": 28.4,
        "esg_score": 82, "env_score": 88, "social_score": 75, "gov_score": 82,
        "scope1_ktco2e": 680, "scope2_ktco2e": 110, "scope3_ktco2e": 4500,
        "revenue_mn": 17200, "renewable_energy_pct": 91,
        "employees": 8100, "women_mgmt_pct": 32, "board_diversity_pct": 40,
        "board_independence_pct": 95,
        "taxonomy_eligible_pct": 94, "taxonomy_aligned_pct": 88,
        "has_sbti": True, "net_zero_year": 2040, "controversy_score": 0,
    },
    {
        "name": "Vestas Wind Systems", "ticker": "VWDRY",
        "sector": "Renewable Energy", "subsector": "Wind Turbines",
        "country": "DK", "market_cap_bn": 18.9,
        "esg_score": 80, "env_score": 86, "social_score": 74, "gov_score": 80,
        "scope1_ktco2e": 55, "scope2_ktco2e": 72, "scope3_ktco2e": 8200,
        "revenue_mn": 15400, "renewable_energy_pct": 78,
        "employees": 28700, "women_mgmt_pct": 24, "board_diversity_pct": 38,
        "board_independence_pct": 91,
        "taxonomy_eligible_pct": 98, "taxonomy_aligned_pct": 91,
        "has_sbti": True, "net_zero_year": 2030, "controversy_score": 1,
    },
    {
        "name": "SolarEdge Technologies", "ticker": "SEDG",
        "sector": "Renewable Energy", "subsector": "Solar Technology",
        "country": "IL", "market_cap_bn": 3.1,
        "esg_score": 63, "env_score": 70, "social_score": 58, "gov_score": 62,
        "scope1_ktco2e": 8, "scope2_ktco2e": 25, "scope3_ktco2e": 650,
        "revenue_mn": 2980, "renewable_energy_pct": 35,
        "employees": 5400, "women_mgmt_pct": 20, "board_diversity_pct": 22,
        "board_independence_pct": 78,
        "taxonomy_eligible_pct": 93, "taxonomy_aligned_pct": 72,
        "has_sbti": False, "net_zero_year": 2050, "controversy_score": 1,
    },
    {
        "name": "Plug Power", "ticker": "PLUG",
        "sector": "Renewable Energy", "subsector": "Hydrogen / Fuel Cells",
        "country": "US", "market_cap_bn": 2.8,
        "esg_score": 55, "env_score": 65, "social_score": 48, "gov_score": 52,
        "scope1_ktco2e": 42, "scope2_ktco2e": 38, "scope3_ktco2e": 280,
        "revenue_mn": 890, "renewable_energy_pct": 45,
        "employees": 3600, "women_mgmt_pct": 18, "board_diversity_pct": 20,
        "board_independence_pct": 75,
        "taxonomy_eligible_pct": 88, "taxonomy_aligned_pct": 55,
        "has_sbti": False, "net_zero_year": 2050, "controversy_score": 2,
    },
    # ── Utilities ─────────────────────────────────────────────────────────
    {
        "name": "Duke Energy", "ticker": "DUK",
        "sector": "Utilities", "subsector": "Electric Utility",
        "country": "US", "market_cap_bn": 82.5,
        "esg_score": 66, "env_score": 60, "social_score": 65, "gov_score": 72,
        "scope1_ktco2e": 54000, "scope2_ktco2e": 1200, "scope3_ktco2e": 8500,
        "revenue_mn": 29100, "renewable_energy_pct": 18,
        "employees": 27600, "women_mgmt_pct": 26, "board_diversity_pct": 35,
        "board_independence_pct": 93,
        "taxonomy_eligible_pct": 45, "taxonomy_aligned_pct": 22,
        "has_sbti": True, "net_zero_year": 2050, "controversy_score": 2,
    },
    {
        "name": "Southern Company", "ticker": "SO",
        "sector": "Utilities", "subsector": "Electric Utility",
        "country": "US", "market_cap_bn": 88.3,
        "esg_score": 62, "env_score": 55, "social_score": 62, "gov_score": 70,
        "scope1_ktco2e": 48000, "scope2_ktco2e": 980, "scope3_ktco2e": 7200,
        "revenue_mn": 24500, "renewable_energy_pct": 15,
        "employees": 27000, "women_mgmt_pct": 24, "board_diversity_pct": 33,
        "board_independence_pct": 92,
        "taxonomy_eligible_pct": 38, "taxonomy_aligned_pct": 18,
        "has_sbti": True, "net_zero_year": 2050, "controversy_score": 2,
    },
    {
        "name": "Dominion Energy", "ticker": "D",
        "sector": "Utilities", "subsector": "Electric Utility",
        "country": "US", "market_cap_bn": 46.2,
        "esg_score": 62, "env_score": 58, "social_score": 60, "gov_score": 68,
        "scope1_ktco2e": 28000, "scope2_ktco2e": 650, "scope3_ktco2e": 4800,
        "revenue_mn": 14400, "renewable_energy_pct": 22,
        "employees": 17200, "women_mgmt_pct": 25, "board_diversity_pct": 30,
        "board_independence_pct": 90,
        "taxonomy_eligible_pct": 48, "taxonomy_aligned_pct": 25,
        "has_sbti": True, "net_zero_year": 2050, "controversy_score": 1,
    },
    {
        "name": "National Grid", "ticker": "NGG",
        "sector": "Utilities", "subsector": "Gas & Electric Utility",
        "country": "UK", "market_cap_bn": 52.8,
        "esg_score": 76, "env_score": 75, "social_score": 72, "gov_score": 80,
        "scope1_ktco2e": 2100, "scope2_ktco2e": 380, "scope3_ktco2e": 26000,
        "revenue_mn": 21600, "renewable_energy_pct": 35,
        "employees": 29300, "women_mgmt_pct": 30, "board_diversity_pct": 42,
        "board_independence_pct": 88,
        "taxonomy_eligible_pct": 62, "taxonomy_aligned_pct": 40,
        "has_sbti": True, "net_zero_year": 2050, "controversy_score": 1,
    },
    {
        "name": "Exelon", "ticker": "EXC",
        "sector": "Utilities", "subsector": "Electric Utility",
        "country": "US", "market_cap_bn": 40.1,
        "esg_score": 69, "env_score": 68, "social_score": 66, "gov_score": 74,
        "scope1_ktco2e": 5200, "scope2_ktco2e": 890, "scope3_ktco2e": 42000,
        "revenue_mn": 21400, "renewable_energy_pct": 55,
        "employees": 19500, "women_mgmt_pct": 27, "board_diversity_pct": 35,
        "board_independence_pct": 91,
        "taxonomy_eligible_pct": 58, "taxonomy_aligned_pct": 35,
        "has_sbti": True, "net_zero_year": 2050, "controversy_score": 1,
    },
    {
        "name": "AES Corporation", "ticker": "AES",
        "sector": "Utilities", "subsector": "Power Generation",
        "country": "US", "market_cap_bn": 12.4,
        "esg_score": 68, "env_score": 70, "social_score": 63, "gov_score": 70,
        "scope1_ktco2e": 32000, "scope2_ktco2e": 520, "scope3_ktco2e": 3500,
        "revenue_mn": 12700, "renewable_energy_pct": 38,
        "employees": 8600, "women_mgmt_pct": 29, "board_diversity_pct": 36,
        "board_independence_pct": 89,
        "taxonomy_eligible_pct": 52, "taxonomy_aligned_pct": 30,
        "has_sbti": True, "net_zero_year": 2040, "controversy_score": 2,
    },
    # ── Oil & Gas — Energy Transition ─────────────────────────────────────
    {
        "name": "TotalEnergies", "ticker": "TTE",
        "sector": "Oil & Gas", "subsector": "Integrated Energy",
        "country": "FR", "market_cap_bn": 148.5,
        "esg_score": 58, "env_score": 52, "social_score": 58, "gov_score": 65,
        "scope1_ktco2e": 38000, "scope2_ktco2e": 4200, "scope3_ktco2e": 380000,
        "revenue_mn": 218700, "renewable_energy_pct": 8,
        "employees": 101000, "women_mgmt_pct": 27, "board_diversity_pct": 42,
        "board_independence_pct": 82,
        "taxonomy_eligible_pct": 18, "taxonomy_aligned_pct": 6,
        "has_sbti": False, "net_zero_year": 2050, "controversy_score": 3,
    },
    {
        "name": "Shell", "ticker": "SHEL",
        "sector": "Oil & Gas", "subsector": "Integrated Energy",
        "country": "NL", "market_cap_bn": 212.3,
        "esg_score": 55, "env_score": 48, "social_score": 55, "gov_score": 62,
        "scope1_ktco2e": 52000, "scope2_ktco2e": 8500, "scope3_ktco2e": 1200000,
        "revenue_mn": 316400, "renewable_energy_pct": 5,
        "employees": 93000, "women_mgmt_pct": 30, "board_diversity_pct": 38,
        "board_independence_pct": 85,
        "taxonomy_eligible_pct": 12, "taxonomy_aligned_pct": 3,
        "has_sbti": False, "net_zero_year": 2050, "controversy_score": 4,
    },
    {
        "name": "BP", "ticker": "BP",
        "sector": "Oil & Gas", "subsector": "Integrated Energy",
        "country": "UK", "market_cap_bn": 96.8,
        "esg_score": 55, "env_score": 50, "social_score": 54, "gov_score": 60,
        "scope1_ktco2e": 34000, "scope2_ktco2e": 5800, "scope3_ktco2e": 320000,
        "revenue_mn": 197300, "renewable_energy_pct": 6,
        "employees": 87800, "women_mgmt_pct": 28, "board_diversity_pct": 36,
        "board_independence_pct": 86,
        "taxonomy_eligible_pct": 14, "taxonomy_aligned_pct": 4,
        "has_sbti": False, "net_zero_year": 2050, "controversy_score": 4,
    },
    {
        "name": "Equinor", "ticker": "EQNR",
        "sector": "Oil & Gas", "subsector": "Integrated Energy",
        "country": "NO", "market_cap_bn": 72.5,
        "esg_score": 66, "env_score": 62, "social_score": 65, "gov_score": 72,
        "scope1_ktco2e": 12500, "scope2_ktco2e": 420, "scope3_ktco2e": 250000,
        "revenue_mn": 105400, "renewable_energy_pct": 12,
        "employees": 22000, "women_mgmt_pct": 35, "board_diversity_pct": 45,
        "board_independence_pct": 80,
        "taxonomy_eligible_pct": 22, "taxonomy_aligned_pct": 10,
        "has_sbti": False, "net_zero_year": 2050, "controversy_score": 2,
    },
    # ── Infrastructure ────────────────────────────────────────────────────
    {
        "name": "Brookfield Infrastructure", "ticker": "BIP",
        "sector": "Infrastructure", "subsector": "Diversified Infrastructure",
        "country": "CA", "market_cap_bn": 15.6,
        "esg_score": 72, "env_score": 72, "social_score": 68, "gov_score": 75,
        "scope1_ktco2e": 4200, "scope2_ktco2e": 1100, "scope3_ktco2e": 8800,
        "revenue_mn": 19800, "renewable_energy_pct": 42,
        "employees": 48000, "women_mgmt_pct": 26, "board_diversity_pct": 30,
        "board_independence_pct": 86,
        "taxonomy_eligible_pct": 55, "taxonomy_aligned_pct": 35,
        "has_sbti": True, "net_zero_year": 2050, "controversy_score": 1,
    },
    {
        "name": "Waste Management", "ticker": "WM",
        "sector": "Infrastructure", "subsector": "Environmental Services",
        "country": "US", "market_cap_bn": 82.3,
        "esg_score": 70, "env_score": 68, "social_score": 67, "gov_score": 75,
        "scope1_ktco2e": 18500, "scope2_ktco2e": 620, "scope3_ktco2e": 5400,
        "revenue_mn": 20400, "renewable_energy_pct": 22,
        "employees": 49000, "women_mgmt_pct": 24, "board_diversity_pct": 33,
        "board_independence_pct": 90,
        "taxonomy_eligible_pct": 42, "taxonomy_aligned_pct": 28,
        "has_sbti": True, "net_zero_year": 2040, "controversy_score": 2,
    },
    # ── Digital Infrastructure ────────────────────────────────────────────
    {
        "name": "Equinix", "ticker": "EQIX",
        "sector": "Digital Infrastructure", "subsector": "Data Centers",
        "country": "US", "market_cap_bn": 78.2,
        "esg_score": 75, "env_score": 78, "social_score": 70, "gov_score": 76,
        "scope1_ktco2e": 85, "scope2_ktco2e": 1450, "scope3_ktco2e": 2100,
        "revenue_mn": 8200, "renewable_energy_pct": 96,
        "employees": 13500, "women_mgmt_pct": 31, "board_diversity_pct": 36,
        "board_independence_pct": 92,
        "taxonomy_eligible_pct": 68, "taxonomy_aligned_pct": 52,
        "has_sbti": True, "net_zero_year": 2030, "controversy_score": 0,
    },
    {
        "name": "Digital Realty", "ticker": "DLR",
        "sector": "Digital Infrastructure", "subsector": "Data Centers",
        "country": "US", "market_cap_bn": 42.1,
        "esg_score": 71, "env_score": 74, "social_score": 67, "gov_score": 73,
        "scope1_ktco2e": 62, "scope2_ktco2e": 1800, "scope3_ktco2e": 1500,
        "revenue_mn": 5500, "renewable_energy_pct": 75,
        "employees": 4200, "women_mgmt_pct": 28, "board_diversity_pct": 33,
        "board_independence_pct": 88,
        "taxonomy_eligible_pct": 65, "taxonomy_aligned_pct": 45,
        "has_sbti": True, "net_zero_year": 2040, "controversy_score": 0,
    },
    {
        "name": "American Tower", "ticker": "AMT",
        "sector": "Digital Infrastructure", "subsector": "Telecom Towers",
        "country": "US", "market_cap_bn": 95.4,
        "esg_score": 69, "env_score": 70, "social_score": 64, "gov_score": 72,
        "scope1_ktco2e": 450, "scope2_ktco2e": 680, "scope3_ktco2e": 1200,
        "revenue_mn": 11100, "renewable_energy_pct": 42,
        "employees": 6200, "women_mgmt_pct": 26, "board_diversity_pct": 30,
        "board_independence_pct": 90,
        "taxonomy_eligible_pct": 55, "taxonomy_aligned_pct": 32,
        "has_sbti": True, "net_zero_year": 2045, "controversy_score": 1,
    },
    {
        "name": "Crown Castle", "ticker": "CCI",
        "sector": "Digital Infrastructure", "subsector": "Telecom Towers",
        "country": "US", "market_cap_bn": 48.7,
        "esg_score": 67, "env_score": 66, "social_score": 64, "gov_score": 70,
        "scope1_ktco2e": 35, "scope2_ktco2e": 290, "scope3_ktco2e": 680,
        "revenue_mn": 6800, "renewable_energy_pct": 30,
        "employees": 5100, "women_mgmt_pct": 23, "board_diversity_pct": 28,
        "board_independence_pct": 85,
        "taxonomy_eligible_pct": 50, "taxonomy_aligned_pct": 28,
        "has_sbti": False, "net_zero_year": 2050, "controversy_score": 1,
    },
    # ── Social Infrastructure ─────────────────────────────────────────────
    {
        "name": "Welltower", "ticker": "WELL",
        "sector": "Social Infrastructure", "subsector": "Healthcare Real Estate",
        "country": "US", "market_cap_bn": 55.6,
        "esg_score": 70, "env_score": 63, "social_score": 74, "gov_score": 72,
        "scope1_ktco2e": 210, "scope2_ktco2e": 580, "scope3_ktco2e": 2800,
        "revenue_mn": 6800, "renewable_energy_pct": 15,
        "employees": 400, "women_mgmt_pct": 35, "board_diversity_pct": 40,
        "board_independence_pct": 91,
        "taxonomy_eligible_pct": 35, "taxonomy_aligned_pct": 18,
        "has_sbti": False, "net_zero_year": 2050, "controversy_score": 0,
    },
]

# ── Emissions Trajectory (Scope 1+2, kt CO2e by year) ───────────────────────

EMISSIONS_TRAJECTORY = {
    "NEE":   {"2020": 28200, "2021": 27100, "2022": 25800, "2023": 24980},
    "FSLR":  {"2020": 520,   "2021": 480,   "2022": 450,   "2023": 435},
    "ENPH":  {"2020": 30,    "2021": 26,    "2022": 24,    "2023": 23},
    "BEPC":  {"2020": 210,   "2021": 190,   "2022": 175,   "2023": 165},
    "DNNGY": {"2020": 1200,  "2021": 980,   "2022": 850,   "2023": 790},
    "VWDRY": {"2020": 165,   "2021": 148,   "2022": 135,   "2023": 127},
    "SEDG":  {"2020": 38,    "2021": 36,    "2022": 34,    "2023": 33},
    "PLUG":  {"2020": 72,    "2021": 74,    "2022": 78,    "2023": 80},
    "DUK":   {"2020": 62000, "2021": 59500, "2022": 57000, "2023": 55200},
    "SO":    {"2020": 56000, "2021": 54000, "2022": 51500, "2023": 48980},
    "D":     {"2020": 33000, "2021": 31200, "2022": 30000, "2023": 28650},
    "NGG":   {"2020": 3100,  "2021": 2800,  "2022": 2600,  "2023": 2480},
    "EXC":   {"2020": 8200,  "2021": 7400,  "2022": 6500,  "2023": 6090},
    "AES":   {"2020": 38000, "2021": 36500, "2022": 34500, "2023": 32520},
    "TTE":   {"2020": 45000, "2021": 44000, "2022": 43000, "2023": 42200},
    "SHEL":  {"2020": 68000, "2021": 65000, "2022": 62000, "2023": 60500},
    "BP":    {"2020": 44000, "2021": 42500, "2022": 41000, "2023": 39800},
    "EQNR":  {"2020": 14500, "2021": 13800, "2022": 13200, "2023": 12920},
    "BIP":   {"2020": 6200,  "2021": 5800,  "2022": 5500,  "2023": 5300},
    "WM":    {"2020": 21000, "2021": 20200, "2022": 19500, "2023": 19120},
    "EQIX":  {"2020": 1900,  "2021": 1750,  "2022": 1600,  "2023": 1535},
    "DLR":   {"2020": 2200,  "2021": 2050,  "2022": 1920,  "2023": 1862},
    "AMT":   {"2020": 1350,  "2021": 1250,  "2022": 1180,  "2023": 1130},
    "CCI":   {"2020": 380,   "2021": 360,   "2022": 340,   "2023": 325},
    "WELL":  {"2020": 950,   "2021": 880,   "2022": 820,   "2023": 790},
}

# ── SFDR Principal Adverse Impact indicator definitions ──────────────────────

PAI_INDICATORS = {
    1:  "GHG Emissions (Scope 1, 2, 3)",
    2:  "Carbon Footprint",
    3:  "GHG Intensity of Investee Companies",
    4:  "Exposure to Companies Active in the Fossil Fuel Sector",
    5:  "Share of Non-Renewable Energy Consumption and Production",
    6:  "Energy Consumption Intensity per High Impact Climate Sector",
    7:  "Activities Negatively Affecting Biodiversity-Sensitive Areas",
    8:  "Emissions to Water",
    9:  "Hazardous Waste and Radioactive Waste Ratio",
    10: "Violations of UN Global Compact / OECD Guidelines",
    11: "Lack of Processes and Compliance Mechanisms to Monitor UNGC/OECD",
    12: "Unadjusted Gender Pay Gap",
    13: "Board Gender Diversity",
    14: "Exposure to Controversial Weapons",
}

# ── EU Taxonomy objectives ───────────────────────────────────────────────────

TAXONOMY_OBJECTIVES = [
    "Climate Change Mitigation",
    "Climate Change Adaptation",
    "Sustainable Use of Water and Marine Resources",
    "Transition to a Circular Economy",
    "Pollution Prevention and Control",
    "Protection of Biodiversity and Ecosystems",
]

SECTORS = sorted(set(c["sector"] for c in COMPANIES))
TICKERS = [c["ticker"] for c in COMPANIES]


def get_esg_dataframe() -> pd.DataFrame:
    """Return the full ESG universe as a DataFrame."""
    df = pd.DataFrame(COMPANIES)
    df["carbon_intensity"] = (
        (df["scope1_ktco2e"] + df["scope2_ktco2e"]) / df["revenue_mn"] * 1000
    )
    return df


def get_emissions_trajectory_df() -> pd.DataFrame:
    """Long-format emissions trajectory for all companies."""
    rows = []
    for ticker, years in EMISSIONS_TRAJECTORY.items():
        for year, val in years.items():
            rows.append({"ticker": ticker, "year": int(year), "emissions_ktco2e": val})
    return pd.DataFrame(rows)
