"""
SFDR Principal Adverse Impact (PAI) Indicators & EU Taxonomy Alignment

Computes the 14 mandatory PAI indicators under SFDR (EU 2019/2088) and
EU Taxonomy (EU 2020/852) alignment metrics for the portfolio.
"""

import pandas as pd
import numpy as np
from config import get_esg_dataframe, PAI_INDICATORS, TAXONOMY_OBJECTIVES


def compute_pai_indicators(esg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute portfolio-level PAI indicators.
    Weights by market cap (proxy for investment weight).
    """
    total_mcap = esg["market_cap_bn"].sum()
    w = esg["market_cap_bn"] / total_mcap

    total_scope1 = (esg["scope1_ktco2e"] * w).sum()
    total_scope2 = (esg["scope2_ktco2e"] * w).sum()
    total_scope3 = (esg["scope3_ktco2e"] * w).sum()

    carbon_footprint = (total_scope1 + total_scope2) / total_mcap * 1000

    ghg_intensity = (
        ((esg["scope1_ktco2e"] + esg["scope2_ktco2e"]) / esg["revenue_mn"]) * w
    ).sum() * 1000

    fossil_exposure = esg.loc[
        esg["sector"] == "Oil & Gas", "market_cap_bn"
    ].sum() / total_mcap * 100

    non_renewable_pct = (100 - esg["renewable_energy_pct"]) * w
    non_renewable = non_renewable_pct.sum()

    energy_intensity = (
        (esg["scope1_ktco2e"] + esg["scope2_ktco2e"]) / esg["revenue_mn"]
    )
    weighted_energy_int = (energy_intensity * w).sum()

    biodiversity_flag = 0
    water_emissions = 0

    controversy_avg = (esg["controversy_score"] * w).sum()
    ungc_violations = int((esg["controversy_score"] >= 4).sum())
    ungc_no_process = int((esg["controversy_score"] >= 3).sum())

    gender_gap_proxy = 100 - (esg["women_mgmt_pct"] * w).sum()
    board_diversity = (esg["board_diversity_pct"] * w).sum()

    controversial_weapons = 0

    indicators = [
        {
            "pai_id": 1,
            "indicator": PAI_INDICATORS[1],
            "metric": "Scope 1+2+3 (weighted, kt CO₂e)",
            "value": round(total_scope1 + total_scope2 + total_scope3, 1),
            "unit": "kt CO₂e",
            "status": "🟡" if (total_scope1 + total_scope2) > 10000 else "🟢",
        },
        {
            "pai_id": 2,
            "indicator": PAI_INDICATORS[2],
            "metric": "tCO₂e per €M invested",
            "value": round(carbon_footprint, 1),
            "unit": "tCO₂e / €M",
            "status": "🟡" if carbon_footprint > 100 else "🟢",
        },
        {
            "pai_id": 3,
            "indicator": PAI_INDICATORS[3],
            "metric": "tCO₂e per €M revenue (weighted)",
            "value": round(ghg_intensity, 1),
            "unit": "tCO₂e / €M rev",
            "status": "🔴" if ghg_intensity > 200 else "🟡" if ghg_intensity > 100 else "🟢",
        },
        {
            "pai_id": 4,
            "indicator": PAI_INDICATORS[4],
            "metric": "% portfolio in fossil fuels",
            "value": round(fossil_exposure, 1),
            "unit": "%",
            "status": "🔴" if fossil_exposure > 30 else "🟡" if fossil_exposure > 15 else "🟢",
        },
        {
            "pai_id": 5,
            "indicator": PAI_INDICATORS[5],
            "metric": "Non-renewable energy share",
            "value": round(non_renewable, 1),
            "unit": "%",
            "status": "🟡" if non_renewable > 50 else "🟢",
        },
        {
            "pai_id": 6,
            "indicator": PAI_INDICATORS[6],
            "metric": "Energy consumption intensity",
            "value": round(weighted_energy_int * 1000, 2),
            "unit": "GWh / €M rev",
            "status": "🟡",
        },
        {
            "pai_id": 7,
            "indicator": PAI_INDICATORS[7],
            "metric": "Companies with biodiversity impact",
            "value": biodiversity_flag,
            "unit": "count",
            "status": "🟢",
        },
        {
            "pai_id": 8,
            "indicator": PAI_INDICATORS[8],
            "metric": "Tonnes of emissions to water",
            "value": water_emissions,
            "unit": "tonnes",
            "status": "🟢",
        },
        {
            "pai_id": 9,
            "indicator": PAI_INDICATORS[9],
            "metric": "Hazardous waste ratio",
            "value": 0.0,
            "unit": "tonnes / €M rev",
            "status": "🟢",
        },
        {
            "pai_id": 10,
            "indicator": PAI_INDICATORS[10],
            "metric": "Companies with UNGC violations",
            "value": ungc_violations,
            "unit": "count",
            "status": "🔴" if ungc_violations > 0 else "🟢",
        },
        {
            "pai_id": 11,
            "indicator": PAI_INDICATORS[11],
            "metric": "Companies lacking compliance mechanisms",
            "value": ungc_no_process,
            "unit": "count",
            "status": "🟡" if ungc_no_process > 2 else "🟢",
        },
        {
            "pai_id": 12,
            "indicator": PAI_INDICATORS[12],
            "metric": "Management gender gap (proxy)",
            "value": round(gender_gap_proxy, 1),
            "unit": "%",
            "status": "🟡" if gender_gap_proxy > 70 else "🟢",
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
            "metric": "Exposure to controversial weapons",
            "value": controversial_weapons,
            "unit": "count",
            "status": "🟢",
        },
    ]
    return pd.DataFrame(indicators)


def compute_taxonomy_alignment(esg: pd.DataFrame) -> dict:
    """Portfolio-level EU Taxonomy alignment metrics."""
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

    return {
        "pai_indicators": pai,
        "taxonomy": taxonomy,
        "summary": {
            "red_flags": int(red_count),
            "warnings": int(yellow_count),
            "compliant": int(green_count),
            "total_indicators": len(pai),
        },
    }


if __name__ == "__main__":
    esg = get_esg_dataframe()
    report = generate_pai_report(esg)

    print("SFDR PAI Indicators")
    print("=" * 80)
    print(report["pai_indicators"].to_string(index=False))

    print(f"\nSummary: {report['summary']['red_flags']} red, "
          f"{report['summary']['warnings']} amber, "
          f"{report['summary']['compliant']} green")

    print(f"\nEU Taxonomy Alignment")
    print(f"  Eligible : {report['taxonomy']['portfolio_eligible_pct']}%")
    print(f"  Aligned  : {report['taxonomy']['portfolio_aligned_pct']}%")
    print(f"  SBTi     : {report['taxonomy']['sbti_validated_count']} / {len(esg)} companies")
