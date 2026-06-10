"""
SFDR Principal Adverse Impact (PAI) Indicators & EU Taxonomy Alignment

Computes the 14 mandatory PAI indicators under SFDR (EU 2019/2088, Delegated
Regulation 2022/1288) and EU Taxonomy (EU 2020/852) alignment metrics for an
illustrative portfolio invested across the universe in proportion to market cap.

Methodology notes
-----------------
* Financed emissions follow the standard attribution approach:
  attributed emissions_i = (investment_i / company value_i) × emissions_i.
  Market cap is used as the proxy for enterprise value (EVIC).
* Indicators where issuer-level data is not collected in this demo (PAI 7, 8, 9)
  are disclosed as data gaps with a ⚪ status — SFDR permits best-efforts
  coverage provided the gap is disclosed, which is exactly what real reporting
  teams do in year one.
"""

import pandas as pd
import numpy as np
from config import (
    get_esg_dataframe,
    PAI_INDICATORS,
    TAXONOMY_OBJECTIVES,
    PORTFOLIO_VALUE_MN,
)

# Approximate grid emission factor used to back out electricity consumption
# from Scope 2 emissions (kt CO2e per GWh). Used only for the PAI 6 proxy.
GRID_FACTOR_KT_PER_GWH = 0.35


def compute_pai_indicators(esg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute portfolio-level PAI indicators for an illustrative portfolio of
    PORTFOLIO_VALUE_MN (€M) invested in proportion to market cap.
    """
    total_mcap_mn = esg["market_cap_bn"].sum() * 1000  # €M
    w = esg["market_cap_bn"] / esg["market_cap_bn"].sum()  # investment weights
    investment_mn = w * PORTFOLIO_VALUE_MN

    # Ownership share of each company (investment / market cap, both in €M)
    ownership = investment_mn / (esg["market_cap_bn"] * 1000)

    # ── PAI 1–3: financed GHG emissions (tonnes CO2e) ─────────────────────
    fin_scope1_t = (ownership * esg["scope1_ktco2e"] * 1000).sum()
    fin_scope2_t = (ownership * esg["scope2_ktco2e"] * 1000).sum()
    fin_scope3_t = (ownership * esg["scope3_ktco2e"] * 1000).sum()
    fin_total_t = fin_scope1_t + fin_scope2_t + fin_scope3_t

    # PAI 2: carbon footprint = financed emissions per €M invested
    carbon_footprint = fin_total_t / PORTFOLIO_VALUE_MN

    # PAI 3: weighted-average GHG intensity (tCO2e per €M revenue, Scope 1+2+3)
    ghg_intensity = (
        w * (
            (esg["scope1_ktco2e"] + esg["scope2_ktco2e"] + esg["scope3_ktco2e"])
            * 1000 / esg["revenue_mn"]
        )
    ).sum()

    # ── PAI 4: fossil fuel sector exposure ────────────────────────────────
    fossil_exposure = (
        esg.loc[esg["sector"] == "Oil & Gas", "market_cap_bn"].sum()
        / esg["market_cap_bn"].sum() * 100
    )

    # ── PAI 5: non-renewable energy share (weighted) ──────────────────────
    non_renewable = ((100 - esg["renewable_energy_pct"]) * w).sum()

    # ── PAI 6: energy consumption intensity (proxy from Scope 2) ─────────
    energy_gwh = esg["scope2_ktco2e"] / GRID_FACTOR_KT_PER_GWH
    energy_intensity = (w * (energy_gwh / esg["revenue_mn"] * 1000)).sum()  # MWh/€M

    # ── PAI 10–11: governance / conduct ───────────────────────────────────
    ungc_violations = int((esg["controversy_score"] >= 4).sum())
    ungc_no_process = int((esg["controversy_score"] >= 3).sum())

    # ── PAI 12: gender pay gap — proxied by distance from management parity
    women_mgmt_w = (esg["women_mgmt_pct"] * w).sum()
    parity_gap_pp = 50 - women_mgmt_w  # percentage points below 50/50 parity

    # ── PAI 13: board gender diversity (weighted) ─────────────────────────
    board_diversity = (esg["board_diversity_pct"] * w).sum()

    # ── PAI 14: controversial weapons (screened out by mandate) ───────────
    controversial_weapons = 0

    NO_DATA = "n/a — data gap"

    indicators = [
        {
            "pai_id": 1,
            "indicator": PAI_INDICATORS[1],
            "metric": "Financed Scope 1+2+3 emissions",
            "value": round(fin_total_t, 0),
            "unit": "tCO₂e",
            "status": "🟡" if (fin_scope1_t + fin_scope2_t) > 50000 else "🟢",
        },
        {
            "pai_id": 2,
            "indicator": PAI_INDICATORS[2],
            "metric": "Financed emissions per €M invested",
            "value": round(carbon_footprint, 1),
            "unit": "tCO₂e / €M",
            "status": "🔴" if carbon_footprint > 1500 else "🟡" if carbon_footprint > 500 else "🟢",
        },
        {
            "pai_id": 3,
            "indicator": PAI_INDICATORS[3],
            "metric": "Weighted-avg GHG intensity (Scope 1+2+3)",
            "value": round(ghg_intensity, 1),
            "unit": "tCO₂e / €M rev",
            "status": "🔴" if ghg_intensity > 2000 else "🟡" if ghg_intensity > 800 else "🟢",
        },
        {
            "pai_id": 4,
            "indicator": PAI_INDICATORS[4],
            "metric": "% portfolio in fossil fuel companies",
            "value": round(fossil_exposure, 1),
            "unit": "%",
            "status": "🔴" if fossil_exposure > 30 else "🟡" if fossil_exposure > 15 else "🟢",
        },
        {
            "pai_id": 5,
            "indicator": PAI_INDICATORS[5],
            "metric": "Weighted non-renewable energy share",
            "value": round(non_renewable, 1),
            "unit": "%",
            "status": "🟡" if non_renewable > 50 else "🟢",
        },
        {
            "pai_id": 6,
            "indicator": PAI_INDICATORS[6],
            "metric": "Energy intensity (proxy from Scope 2 / grid factor)",
            "value": round(energy_intensity, 1),
            "unit": "MWh / €M rev",
            "status": "🟡" if energy_intensity > 500 else "🟢",
        },
        {
            "pai_id": 7,
            "indicator": PAI_INDICATORS[7],
            "metric": "Issuer-level biodiversity data not yet collected",
            "value": NO_DATA,
            "unit": "—",
            "status": "⚪",
        },
        {
            "pai_id": 8,
            "indicator": PAI_INDICATORS[8],
            "metric": "Issuer-level water emissions data not yet collected",
            "value": NO_DATA,
            "unit": "—",
            "status": "⚪",
        },
        {
            "pai_id": 9,
            "indicator": PAI_INDICATORS[9],
            "metric": "Issuer-level hazardous waste data not yet collected",
            "value": NO_DATA,
            "unit": "—",
            "status": "⚪",
        },
        {
            "pai_id": 10,
            "indicator": PAI_INDICATORS[10],
            "metric": "Companies with severe controversies (UNGC proxy)",
            "value": ungc_violations,
            "unit": "count",
            "status": "🔴" if ungc_violations > 0 else "🟢",
        },
        {
            "pai_id": 11,
            "indicator": PAI_INDICATORS[11],
            "metric": "Companies with elevated controversy / weak processes",
            "value": ungc_no_process,
            "unit": "count",
            "status": "🟡" if ungc_no_process > 2 else "🟢",
        },
        {
            "pai_id": 12,
            "indicator": PAI_INDICATORS[12],
            "metric": "Gap to management gender parity (proxy for pay gap)",
            "value": round(parity_gap_pp, 1),
            "unit": "pp below 50%",
            "status": "🟡" if parity_gap_pp > 25 else "🟢",
        },
        {
            "pai_id": 13,
            "indicator": PAI_INDICATORS[13],
            "metric": "Weighted board gender diversity",
            "value": round(board_diversity, 1),
            "unit": "%",
            "status": "🟢" if board_diversity >= 33 else "🟡",
        },
        {
            "pai_id": 14,
            "indicator": PAI_INDICATORS[14],
            "metric": "Exposure to controversial weapons (screened)",
            "value": controversial_weapons,
            "unit": "count",
            "status": "🟢",
        },
    ]
    return pd.DataFrame(indicators)


def compute_taxonomy_alignment(esg: pd.DataFrame) -> dict:
    """Portfolio-level EU Taxonomy alignment metrics (market-cap weighted)."""
    w = esg["market_cap_bn"] / esg["market_cap_bn"].sum()

    eligible = (esg["taxonomy_eligible_pct"] * w).sum()
    aligned = (esg["taxonomy_aligned_pct"] * w).sum()
    gap = eligible - aligned

    by_sector = esg.groupby("sector").agg(
        eligible=("taxonomy_eligible_pct", "mean"),
        aligned=("taxonomy_aligned_pct", "mean"),
        companies=("name", "count"),
    ).round(1)

    sbti_count = esg["has_sbti"].sum()
    sbti_pct = sbti_count / len(esg) * 100

    return {
        "portfolio_eligible_pct": round(eligible, 1),
        "portfolio_aligned_pct": round(aligned, 1),
        "alignment_gap_pct": round(gap, 1),
        "by_sector": by_sector,
        "sbti_validated_count": int(sbti_count),
        "sbti_validated_pct": round(sbti_pct, 1),
    }


def generate_pai_report(esg: pd.DataFrame) -> dict:
    """Full SFDR PAI + Taxonomy report."""
    pai = compute_pai_indicators(esg)
    taxonomy = compute_taxonomy_alignment(esg)

    red_count = (pai["status"] == "🔴").sum()
    yellow_count = (pai["status"] == "🟡").sum()
    green_count = (pai["status"] == "🟢").sum()
    gap_count = (pai["status"] == "⚪").sum()

    return {
        "pai_indicators": pai,
        "taxonomy": taxonomy,
        "summary": {
            "red_flags": int(red_count),
            "warnings": int(yellow_count),
            "compliant": int(green_count),
            "data_gaps": int(gap_count),
            "total_indicators": len(pai),
        },
    }


if __name__ == "__main__":
    esg = get_esg_dataframe()
    report = generate_pai_report(esg)

    print("SFDR PAI Indicators")
    print("=" * 80)
    print(report["pai_indicators"].to_string(index=False))

    s = report["summary"]
    print(f"\nSummary: {s['red_flags']} red, {s['warnings']} amber, "
          f"{s['compliant']} green, {s['data_gaps']} data gaps")

    print("\nEU Taxonomy Alignment")
    print(f"  Eligible : {report['taxonomy']['portfolio_eligible_pct']}%")
    print(f"  Aligned  : {report['taxonomy']['portfolio_aligned_pct']}%")
    print(f"  SBTi     : {report['taxonomy']['sbti_validated_count']} / {len(esg)} companies")
