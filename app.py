"""
InfraESG Analytics — Interactive Streamlit Dashboard
ESG Risk Assessment & Decarbonisation Tracking for Infrastructure Investments
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import get_esg_dataframe, get_emissions_trajectory_df, SECTORS, PAI_INDICATORS
from sfdr_pai import generate_pai_report, compute_taxonomy_alignment
from analysis import (
    cluster_companies, sector_benchmarks,
    train_models, get_feature_importance, FEATURE_COLS, TARGET_COL,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="InfraESG Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "green": "#2E7D32", "light_green": "#A5D6A7",
    "blue": "#1565C0", "amber": "#FF8F00",
    "red": "#C62828", "purple": "#6A1B9A",
    "grey": "#616161",
}

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    esg = get_esg_dataframe()
    esg = cluster_companies(esg)
    trajectory = get_emissions_trajectory_df()

    panel = None
    panel_path = "data/panel_data.csv"
    if os.path.exists(panel_path):
        panel = pd.read_csv(panel_path)

    return esg, trajectory, panel


esg, trajectory, panel = load_data()
pai_report = generate_pai_report(esg)
taxonomy = compute_taxonomy_alignment(esg)
benchmarks = sector_benchmarks(esg)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/deciduous-tree.png", width=60)
    st.title("InfraESG Analytics")
    st.caption("ESG Risk Assessment & Decarbonisation Tracking for Infrastructure Investments")
    st.divider()

    sector_filter = st.multiselect(
        "Filter by Sector",
        options=SECTORS,
        default=SECTORS,
    )

    st.divider()
    st.markdown(
        "**Data sources:** Company sustainability reports, "
        "CDP disclosures, Yahoo Finance, proxy statements."
    )
    st.markdown("Built by **Mridul Daga**")

esg_filtered = esg[esg["sector"].isin(sector_filter)]

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary",
    "🏢 ESG Portfolio",
    "🌍 Decarbonisation",
    "📋 SFDR PAI",
    "🤖 ESG Risk Model",
    "🔍 Due Diligence",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Portfolio Executive Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Companies", len(esg_filtered))
    c2.metric("Avg ESG Score", f"{esg_filtered['esg_score'].mean():.1f}")
    c3.metric(
        "Carbon Intensity",
        f"{esg_filtered['carbon_intensity'].mean():.0f}",
        help="tCO₂e per €M revenue (weighted average)",
    )
    c4.metric("Taxonomy Aligned", f"{taxonomy['portfolio_aligned_pct']}%")
    c5.metric("SBTi Validated", f"{taxonomy['sbti_validated_count']}/{len(esg)}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("ESG Score Distribution")
        fig = px.histogram(
            esg_filtered, x="esg_score", nbins=12, color="sector",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"esg_score": "ESG Score", "sector": "Sector"},
        )
        fig.update_layout(bargap=0.1, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Sector Average Scores")
        bench = benchmarks.reset_index()
        fig = go.Figure()
        for col, color, label in [
            ("env_score", COLORS["green"], "Environmental"),
            ("social_score", COLORS["blue"], "Social"),
            ("gov_score", COLORS["purple"], "Governance"),
        ]:
            fig.add_trace(go.Bar(
                x=bench["sector"], y=bench[col], name=label,
                marker_color=color,
            ))
        fig.update_layout(barmode="group", height=350, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Portfolio ESG Heatmap")
    heat_data = esg_filtered.set_index("name")[["env_score", "social_score", "gov_score"]].sort_values("env_score", ascending=False)
    fig = px.imshow(
        heat_data.values,
        x=["Environmental", "Social", "Governance"],
        y=heat_data.index,
        color_continuous_scale="RdYlGn",
        zmin=40, zmax=90,
        aspect="auto",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ESG Portfolio
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("ESG Portfolio Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ESG Scores by Company")
        fig = px.bar(
            esg_filtered.sort_values("esg_score", ascending=True),
            x="esg_score", y="name", orientation="h",
            color="esg_profile",
            color_discrete_map={
                "ESG Leader": COLORS["green"],
                "Strong Performer": COLORS["blue"],
                "Transitioning": COLORS["amber"],
                "Laggard": COLORS["red"],
            },
            labels={"esg_score": "ESG Score", "name": "", "esg_profile": "Profile"},
        )
        fig.update_layout(height=600, yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ESG Profile Radar")
        selected = st.selectbox("Select company", esg_filtered["name"].tolist(), key="radar_company")
        row = esg_filtered[esg_filtered["name"] == selected].iloc[0]

        categories = ["Environmental", "Social", "Governance", "Taxonomy", "Renewables"]
        values = [
            row["env_score"], row["social_score"], row["gov_score"],
            row["taxonomy_aligned_pct"], row["renewable_energy_pct"],
        ]
        values.append(values[0])
        categories.append(categories[0])

        fig = go.Figure(go.Scatterpolar(
            r=values, theta=categories, fill="toself",
            line_color=COLORS["green"], fillcolor="rgba(46,125,50,0.2)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=400, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **ESG Score** | {row['esg_score']} |
        | **Carbon Intensity** | {row['carbon_intensity']:.0f} tCO₂e/€M |
        | **Renewable Energy** | {row['renewable_energy_pct']}% |
        | **EU Taxonomy Aligned** | {row['taxonomy_aligned_pct']}% |
        | **SBTi Validated** | {'Yes' if row['has_sbti'] else 'No'} |
        | **Net Zero Target** | {row['net_zero_year']} |
        | **Profile** | {row['esg_profile']} |
        """)

    st.divider()
    st.subheader("Carbon Intensity vs ESG Score")
    fig = px.scatter(
        esg_filtered, x="esg_score", y="carbon_intensity",
        size="market_cap_bn", color="sector",
        hover_name="name",
        labels={
            "esg_score": "ESG Score",
            "carbon_intensity": "Carbon Intensity (tCO₂e/€M)",
            "market_cap_bn": "Market Cap ($B)",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Decarbonisation Tracker
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Decarbonisation Tracker")

    traj_merged = trajectory.merge(esg[["ticker", "name", "sector", "has_sbti"]], on="ticker")
    traj_filtered = traj_merged[traj_merged["sector"].isin(sector_filter)]

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Sector Emissions Trajectories")
        sector_agg = traj_filtered.groupby(["sector", "year"])["emissions_ktco2e"].sum().reset_index()
        base = sector_agg[sector_agg["year"] == 2020].set_index("sector")["emissions_ktco2e"]
        sector_agg["indexed"] = sector_agg.apply(
            lambda r: r["emissions_ktco2e"] / base.get(r["sector"], 1) * 100, axis=1,
        )
        fig = px.line(
            sector_agg, x="year", y="indexed", color="sector",
            markers=True,
            labels={"indexed": "Emissions (2020 = 100)", "year": "Year"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.add_hline(y=50, line_dash="dash", line_color="red",
                      annotation_text="Paris-aligned 2030 target (−50%)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Company-Level Reductions (2020→2023)")
        pivot = traj_filtered.pivot_table(
            index=["ticker", "name", "sector", "has_sbti"],
            columns="year", values="emissions_ktco2e",
        ).reset_index()
        if 2020 in pivot.columns and 2023 in pivot.columns:
            pivot["reduction_pct"] = round((1 - pivot[2023] / pivot[2020]) * 100, 1)
            pivot_sorted = pivot.sort_values("reduction_pct", ascending=False)
            fig = px.bar(
                pivot_sorted, x="reduction_pct", y="name", orientation="h",
                color="has_sbti",
                color_discrete_map={True: COLORS["green"], False: COLORS["grey"]},
                labels={"reduction_pct": "Emissions Reduction (%)", "name": "", "has_sbti": "SBTi Validated"},
            )
            fig.update_layout(height=550, yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("SBTi Target Coverage")
    sbti_df = esg_filtered[["name", "sector", "has_sbti", "net_zero_year"]].copy()
    sbti_df["SBTi Status"] = sbti_df["has_sbti"].map({True: "Validated", False: "Not Validated"})

    c1, c2 = st.columns(2)
    with c1:
        sbti_counts = sbti_df["SBTi Status"].value_counts()
        fig = px.pie(
            values=sbti_counts.values, names=sbti_counts.index,
            color_discrete_map={"Validated": COLORS["green"], "Not Validated": COLORS["grey"]},
            hole=0.4,
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        nz = esg_filtered.groupby("net_zero_year").size().reset_index(name="count")
        fig = px.bar(nz, x="net_zero_year", y="count", labels={"net_zero_year": "Net Zero Target Year", "count": "Companies"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SFDR PAI Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("SFDR Principal Adverse Impact Indicators")
    st.caption("Mandatory PAI disclosure under SFDR (EU 2019/2088)")

    pai = pai_report["pai_indicators"]
    summary = pai_report["summary"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Red Flags", summary["red_flags"], help="Requires immediate attention")
    c2.metric("Warnings", summary["warnings"], help="Monitor and improve")
    c3.metric("Compliant", summary["compliant"], help="Meets thresholds")

    st.divider()

    for _, row in pai.iterrows():
        with st.expander(f"{row['status']} PAI {row['pai_id']}: {row['indicator']}", expanded=row['status'] == '🔴'):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(f"**Metric:** {row['metric']}")
            c2.write(f"**Value:** {row['value']}")
            c3.write(f"**Unit:** {row['unit']}")

    st.divider()
    st.subheader("EU Taxonomy Alignment")

    c1, c2, c3 = st.columns(3)
    c1.metric("Taxonomy Eligible", f"{taxonomy['portfolio_eligible_pct']}%")
    c2.metric("Taxonomy Aligned", f"{taxonomy['portfolio_aligned_pct']}%")
    c3.metric("Alignment Gap", f"{taxonomy['alignment_gap_pct']}%")

    st.subheader("Taxonomy Alignment by Sector")
    tax_sector = taxonomy["by_sector"].reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tax_sector["sector"], y=tax_sector["eligible"], name="Eligible", marker_color=COLORS["light_green"]))
    fig.add_trace(go.Bar(x=tax_sector["sector"], y=tax_sector["aligned"], name="Aligned", marker_color=COLORS["green"]))
    fig.update_layout(barmode="group", height=350, yaxis_title="Percentage (%)")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ESG Risk Model (ML)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("ESG-Driven Volatility Risk Model")

    if panel is not None and TARGET_COL in panel.columns:
        model_output = train_models(panel)
        results = model_output["results"]

        st.subheader("Model Comparison")
        comp_rows = []
        for name, res in results.items():
            comp_rows.append({
                "Model": name,
                "RMSE": round(res["rmse"], 4),
                "MAE": round(res["mae"], 4),
                "R²": round(res["r2"], 3),
                "CV-RMSE": round(res["cv_rmse"], 4),
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feature Importance")
            imp = get_feature_importance(model_output)
            fig = px.bar(
                imp, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Greens",
                labels={"importance": "Importance", "feature": ""},
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Actual vs Predicted Volatility")
            best = results["Random Forest"]
            pred_df = pd.DataFrame({"Actual": model_output["y"], "Predicted": best["predictions"]})
            fig = px.scatter(pred_df, x="Actual", y="Predicted", opacity=0.5)
            fig.add_trace(go.Scatter(
                x=[pred_df["Actual"].min(), pred_df["Actual"].max()],
                y=[pred_df["Actual"].min(), pred_df["Actual"].max()],
                mode="lines", line=dict(dash="dash", color="red"),
                name="Perfect",
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Interactive Volatility Predictor")
        st.caption("Adjust ESG parameters to predict portfolio volatility")

        pc1, pc2, pc3 = st.columns(3)
        esg_input = pc1.slider("ESG Score", 30, 95, 70)
        env_input = pc1.slider("Environmental Score", 30, 95, 70)
        soc_input = pc2.slider("Social Score", 30, 95, 65)
        gov_input = pc2.slider("Governance Score", 30, 95, 70)
        ci_input = pc3.slider("Carbon Intensity", 0, 2000, 500)
        ren_input = pc3.slider("Renewable Energy %", 0, 100, 50)
        tax_input = pc1.slider("Taxonomy Aligned %", 0, 100, 40)
        bd_input = pc2.slider("Board Diversity %", 10, 60, 33)
        bi_input = pc3.slider("Board Independence %", 50, 100, 85)

        if st.button("Predict Volatility", type="primary"):
            input_arr = np.array([[
                esg_input, env_input, soc_input, gov_input,
                ci_input, ren_input, tax_input, bd_input, bi_input,
            ]])
            rf = results["Random Forest"]["model"]
            pred = rf.predict(input_arr)[0]
            risk_level = "Low" if pred < 0.25 else "Medium" if pred < 0.40 else "High"
            color = COLORS["green"] if risk_level == "Low" else COLORS["amber"] if risk_level == "Medium" else COLORS["red"]
            st.markdown(
                f"### Predicted Annualised Volatility: "
                f"<span style='color:{color}'>{pred:.4f} ({risk_level} Risk)</span>",
                unsafe_allow_html=True,
            )

    else:
        st.info(
            "Run `python data_collection.py` to download stock data "
            "and enable the ML risk model. This requires internet access."
        )
        st.markdown("""
        **Model Architecture:**
        - **Features:** ESG scores (E/S/G), carbon intensity, renewable energy %, 
          taxonomy alignment, board diversity & independence
        - **Target:** Annualised realised volatility (rolling 21-day, √252 scaled)
        - **Models:** Linear Regression, Random Forest, Gradient Boosting
        - **Validation:** TimeSeriesSplit 5-fold cross-validation
        - **Dataset:** ~1,200 company-month observations (25 companies × ~48 months)
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Due Diligence Tool
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("ESG Due Diligence Scorecard")
    st.caption("Select an infrastructure company for full ESG assessment")

    company = st.selectbox("Select Company", esg_filtered["name"].tolist(), key="dd_company")
    co = esg_filtered[esg_filtered["name"] == company].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("ESG Scoring")
        for label, field, threshold in [
            ("Overall ESG", "esg_score", 65),
            ("Environmental", "env_score", 65),
            ("Social", "social_score", 60),
            ("Governance", "gov_score", 65),
        ]:
            val = co[field]
            delta = f"{val - threshold:+d} vs threshold"
            st.metric(label, val, delta)

    with col2:
        st.subheader("Climate & Energy")
        st.metric("Carbon Intensity", f"{co['carbon_intensity']:.0f} tCO₂e/€M")
        st.metric("Renewable Energy", f"{co['renewable_energy_pct']}%")
        st.metric("EU Taxonomy Aligned", f"{co['taxonomy_aligned_pct']}%")
        st.metric("SBTi Validated", "Yes" if co["has_sbti"] else "No")
        st.metric("Net Zero Target", co["net_zero_year"])

    with col3:
        st.subheader("Social & Governance")
        st.metric("Board Gender Diversity", f"{co['board_diversity_pct']}%")
        st.metric("Board Independence", f"{co['board_independence_pct']}%")
        st.metric("Women in Management", f"{co['women_mgmt_pct']}%")
        st.metric("Controversy Score", f"{co['controversy_score']} / 5")
        st.metric("Employees", f"{co['employees']:,}")

    st.divider()

    st.subheader("Emissions Trajectory")
    co_traj = trajectory[trajectory["ticker"] == co["ticker"]]
    if not co_traj.empty:
        fig = px.line(
            co_traj, x="year", y="emissions_ktco2e", markers=True,
            labels={"emissions_ktco2e": "Scope 1+2 Emissions (kt CO₂e)", "year": "Year"},
        )
        fig.update_traces(line_color=COLORS["green"], line_width=3)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Risk Assessment Summary")
    risks = []
    if co["esg_score"] < 60:
        risks.append(("🔴", "Low overall ESG score — below investment threshold"))
    if co["controversy_score"] >= 3:
        risks.append(("🔴", "Elevated controversy risk — due diligence escalation required"))
    if co["taxonomy_aligned_pct"] < 20:
        risks.append(("🟡", "Low EU Taxonomy alignment — limited SFDR Article 9 eligibility"))
    if not co["has_sbti"]:
        risks.append(("🟡", "No SBTi-validated target — decarbonisation commitment unverified"))
    if co["board_diversity_pct"] < 30:
        risks.append(("🟡", "Board gender diversity below 30% threshold"))
    if co["renewable_energy_pct"] < 20:
        risks.append(("🟡", "Low renewable energy share — transition risk exposure"))
    if co["carbon_intensity"] > 500:
        risks.append(("🟡", "High carbon intensity — stranded asset risk"))
    if not risks:
        risks.append(("🟢", "No material ESG risks identified"))

    for icon, msg in risks:
        st.markdown(f"{icon} {msg}")

    invest_score = (
        co["esg_score"] * 0.3
        + co["taxonomy_aligned_pct"] * 0.2
        + co["renewable_energy_pct"] * 0.15
        + co["board_diversity_pct"] * 0.1
        + co["board_independence_pct"] * 0.1
        + (100 - co["controversy_score"] * 20) * 0.15
    )
    st.metric(
        "ESG Investment Readiness Score",
        f"{invest_score:.0f} / 100",
        help="Composite score: ESG (30%), Taxonomy (20%), Renewables (15%), "
             "Controversy (15%), Board Diversity (10%), Independence (10%)",
    )

# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "InfraESG Analytics | Data: Company sustainability reports, CDP, Yahoo Finance | "
    "Built by Mridul Daga"
)
