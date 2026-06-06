"""
ESG Analysis Engine
Correlation analysis, ML-based volatility prediction, feature importance,
and ESG-profile clustering for infrastructure companies.
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_esg_dataframe, get_emissions_trajectory_df, SECTORS

warnings.filterwarnings("ignore")
CHART_DIR = "data/charts"
os.makedirs(CHART_DIR, exist_ok=True)


# ── Correlation Analysis ─────────────────────────────────────────────────────

def compute_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlations between ESG and financial metrics."""
    cols = [
        "esg_score", "env_score", "social_score", "gov_score",
        "annualised_vol", "annualised_return", "sharpe_ratio",
        "carbon_intensity", "renewable_energy_pct", "taxonomy_aligned_pct",
    ]
    available = [c for c in cols if c in panel.columns]
    return panel[available].corr()


def plot_correlation_matrix(corr: pd.DataFrame, path: str | None = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, ax=ax, square=True,
        linewidths=0.5,
    )
    ax.set_title("ESG–Financial Correlation Matrix", fontsize=14, pad=15)
    plt.tight_layout()
    dest = path or os.path.join(CHART_DIR, "correlation_matrix.png")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


# ── ML Volatility Prediction ─────────────────────────────────────────────────

FEATURE_COLS = [
    "esg_score", "env_score", "social_score", "gov_score",
    "carbon_intensity", "renewable_energy_pct", "taxonomy_aligned_pct",
    "board_diversity_pct", "board_independence_pct",
]

TARGET_COL = "annualised_vol"


def train_models(panel: pd.DataFrame) -> dict:
    """Train multiple regressors and return comparison metrics."""
    df = panel.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
        ),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    for name, model in models.items():
        X_in = X_scaled if name == "Linear Regression" else X.values
        cv_scores = cross_val_score(model, X_in, y, cv=tscv, scoring="neg_root_mean_squared_error")
        model.fit(X_in, y)
        y_pred = model.predict(X_in)
        results[name] = {
            "model": model,
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "cv_rmse": -cv_scores.mean(),
            "cv_rmse_std": cv_scores.std(),
            "predictions": y_pred,
        }

    return {
        "results": results,
        "X": X, "y": y,
        "feature_names": FEATURE_COLS,
        "scaler": scaler,
    }


def get_feature_importance(model_output: dict, model_name: str = "Random Forest") -> pd.DataFrame:
    """Extract feature importances from a tree-based model."""
    model = model_output["results"][model_name]["model"]
    imp = pd.DataFrame({
        "feature": model_output["feature_names"],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp


def plot_feature_importance(imp: pd.DataFrame, path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("YlGn_r", len(imp))
    ax.barh(imp["feature"], imp["importance"], color=colors)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — Volatility Prediction", fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    dest = path or os.path.join(CHART_DIR, "feature_importance.png")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def plot_actual_vs_predicted(y_true, y_pred, path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="k", linewidths=0.3)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Annualised Volatility")
    ax.set_ylabel("Predicted Annualised Volatility")
    ax.set_title("Actual vs Predicted Volatility")
    ax.legend()
    plt.tight_layout()
    dest = path or os.path.join(CHART_DIR, "actual_vs_predicted.png")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


# ── ESG Profile Clustering ───────────────────────────────────────────────────

def cluster_companies(esg: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """K-means clustering on ESG sub-scores."""
    cluster_features = ["env_score", "social_score", "gov_score"]
    X = esg[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    esg = esg.copy()
    esg["esg_cluster"] = km.fit_predict(X_scaled)

    label_map = {}
    for cid in range(n_clusters):
        mask = esg["esg_cluster"] == cid
        avg = esg.loc[mask, "esg_score"].mean()
        label_map[cid] = avg
    rank = sorted(label_map, key=lambda k: label_map[k], reverse=True)
    names = ["ESG Leader", "Strong Performer", "Transitioning", "Laggard"]
    name_map = {rank[i]: names[min(i, len(names) - 1)] for i in range(len(rank))}
    esg["esg_profile"] = esg["esg_cluster"].map(name_map)

    return esg


# ── Sector Benchmarking ──────────────────────────────────────────────────────

def sector_benchmarks(esg: pd.DataFrame) -> pd.DataFrame:
    """Average ESG metrics by sector."""
    agg_cols = [
        "esg_score", "env_score", "social_score", "gov_score",
        "carbon_intensity", "renewable_energy_pct", "taxonomy_aligned_pct",
        "board_diversity_pct",
    ]
    available = [c for c in agg_cols if c in esg.columns]
    return esg.groupby("sector")[available].mean().round(1)


def plot_sector_esg(benchmarks: pd.DataFrame, path: str | None = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    benchmarks[["env_score", "social_score", "gov_score"]].plot(
        kind="bar", ax=ax, color=["#2e7d32", "#1565c0", "#6a1b9a"],
    )
    ax.set_ylabel("Score (0–100)")
    ax.set_title("Average E / S / G Scores by Sector")
    ax.legend(["Environmental", "Social", "Governance"])
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    dest = path or os.path.join(CHART_DIR, "sector_esg.png")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


# ── Decarbonisation Charts ───────────────────────────────────────────────────

def plot_decarbonisation(trajectory: pd.DataFrame, esg: pd.DataFrame, path: str | None = None):
    """Plot emissions reduction trajectories by sector."""
    merged = trajectory.merge(esg[["ticker", "sector", "name"]], on="ticker")
    fig, ax = plt.subplots(figsize=(10, 6))

    for sector in merged["sector"].unique():
        sub = merged[merged["sector"] == sector]
        agg = sub.groupby("year")["emissions_ktco2e"].sum()
        base = agg.iloc[0]
        pct = (agg / base) * 100
        ax.plot(pct.index, pct.values, marker="o", label=sector, linewidth=2)

    ax.axhline(y=50, color="red", linestyle="--", alpha=0.6, label="Paris-aligned 2030 target (−50 %)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Emissions (indexed to 2020 = 100)")
    ax.set_title("Sector Decarbonisation Trajectories")
    ax.legend(fontsize=8)
    plt.tight_layout()
    dest = path or os.path.join(CHART_DIR, "decarbonisation.png")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


# ── Convenience runner ───────────────────────────────────────────────────────

def run_full_analysis(panel: pd.DataFrame | None = None) -> dict:
    """Execute all analyses. If panel is None, use ESG-only mode."""
    esg = get_esg_dataframe()
    trajectory = get_emissions_trajectory_df()

    out: dict = {}

    esg_clustered = cluster_companies(esg)
    out["esg"] = esg_clustered
    out["benchmarks"] = sector_benchmarks(esg)
    out["trajectory"] = trajectory

    plot_sector_esg(out["benchmarks"])
    plot_decarbonisation(trajectory, esg)

    if panel is not None and TARGET_COL in panel.columns:
        corr = compute_correlations(panel)
        out["correlation"] = corr
        plot_correlation_matrix(corr)

        model_out = train_models(panel)
        out["models"] = model_out

        imp = get_feature_importance(model_out)
        out["feature_importance"] = imp
        plot_feature_importance(imp)

        best = model_out["results"]["Random Forest"]
        plot_actual_vs_predicted(model_out["y"], best["predictions"])

    return out


if __name__ == "__main__":
    panel_path = "data/panel_data.csv"
    panel = None
    if os.path.exists(panel_path):
        panel = pd.read_csv(panel_path)
        print(f"Loaded panel data: {panel.shape}")

    out = run_full_analysis(panel)
    print("\nSector benchmarks:")
    print(out["benchmarks"])

    if "models" in out:
        print("\nModel comparison:")
        for name, res in out["models"]["results"].items():
            print(f"  {name:25s}  RMSE={res['rmse']:.4f}  R²={res['r2']:.3f}  CV-RMSE={res['cv_rmse']:.4f}")
