"""
Data Collection Pipeline
Downloads stock prices via yfinance and assembles a panel dataset
(company × month) merging financial metrics with ESG attributes.
"""

import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from config import COMPANIES, TICKERS, get_esg_dataframe

warnings.filterwarnings("ignore")

DATA_DIR = "data"
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"


def download_stock_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily adjusted close prices for the universe."""
    print(f"Downloading prices for {len(tickers)} tickers …")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=0)
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    return prices


def compute_monthly_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """From daily prices, compute monthly return, realised vol, and Sharpe."""
    daily_ret = prices.pct_change().dropna()

    records = []
    for ticker in prices.columns:
        ts = daily_ret[ticker].dropna()
        monthly = ts.resample("ME").agg(["mean", "std", "count"])
        for date, row in monthly.iterrows():
            if row["count"] < 10:
                continue
            ann_vol = row["std"] * np.sqrt(252)
            ann_ret = row["mean"] * 252
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            records.append({
                "ticker": ticker,
                "date": date,
                "monthly_return": row["mean"] * row["count"],
                "annualised_vol": ann_vol,
                "annualised_return": ann_ret,
                "sharpe_ratio": sharpe,
            })
    return pd.DataFrame(records)


def download_vix() -> pd.DataFrame:
    """Download VIX index as a market-risk proxy."""
    vix = yf.download("^VIX", start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix_close = vix.xs("Close", axis=1, level=0)
    else:
        vix_close = vix[["Close"]]
    vix_monthly = vix_close.resample("ME").mean()
    vix_monthly.columns = ["vix"]
    return vix_monthly


def build_panel_dataset() -> pd.DataFrame:
    """Merge financial metrics with ESG attributes into a panel dataset."""
    prices = download_stock_prices(TICKERS)
    prices.to_csv(os.path.join(DATA_DIR, "stock_prices.csv"))
    print(f"  → Saved daily prices ({prices.shape[0]} days × {prices.shape[1]} tickers)")

    monthly = compute_monthly_metrics(prices)

    vix = download_vix()
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly = monthly.merge(
        vix.reset_index().rename(columns={"Date": "date", "index": "date"}),
        on="date", how="left",
    )
    if "vix" not in monthly.columns:
        monthly["vix"] = np.nan

    esg = get_esg_dataframe()
    panel = monthly.merge(esg, on="ticker", how="left")

    panel.to_csv(os.path.join(DATA_DIR, "panel_data.csv"), index=False)
    print(f"  → Saved panel dataset ({panel.shape[0]} rows × {panel.shape[1]} cols)")

    esg.to_csv(os.path.join(DATA_DIR, "esg_universe.csv"), index=False)
    print(f"  → Saved ESG universe ({esg.shape[0]} companies)")

    return panel


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "charts"), exist_ok=True)

    panel = build_panel_dataset()

    print("\n── Summary ─────────────────────────────────────")
    print(f"Companies : {panel['ticker'].nunique()}")
    print(f"Months    : {panel['date'].nunique()}")
    print(f"Rows      : {len(panel)}")
    print(f"Columns   : {list(panel.columns)}")
    print("Done.")
