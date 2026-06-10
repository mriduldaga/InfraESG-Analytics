"""
Data Collection Pipeline
Downloads stock prices via yfinance and assembles a panel dataset
(company × month) merging financial metrics with ESG attributes.
"""

import os
import warnings
from datetime import date

import pandas as pd
import numpy as np
import yfinance as yf

from config import COMPANIES, TICKERS, get_esg_dataframe

warnings.filterwarnings("ignore")

DATA_DIR = "data"
START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()  # always pull through the latest close
MIN_TRADING_DAYS_PER_MONTH = 10


def download_stock_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily close prices for the universe, dropping failed tickers."""
    print(f"Downloading prices for {len(tickers)} tickers …")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=0)
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)

    # Drop tickers that failed entirely (all-NaN columns) and report them
    failed = [t for t in prices.columns if prices[t].isna().all()]
    if failed:
        print(f"  ⚠ No data for: {', '.join(failed)} — excluded from panel")
        prices = prices.drop(columns=failed)

    return prices


def compute_monthly_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """From daily prices, compute monthly compounded return, realised vol, Sharpe."""
    records = []
    for ticker in prices.columns:
        # Per-ticker returns so one company's gaps don't delete everyone's rows
        ts = prices[ticker].dropna().pct_change().dropna()
        if ts.empty:
            continue
        monthly = ts.resample("ME").agg(["mean", "std", "count"])
        compounded = ts.resample("ME").apply(lambda r: (1 + r).prod() - 1)
        for dt, row in monthly.iterrows():
            if row["count"] < MIN_TRADING_DAYS_PER_MONTH or pd.isna(row["std"]):
                continue
            ann_vol = row["std"] * np.sqrt(252)
            ann_ret = row["mean"] * 252
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            records.append({
                "ticker": ticker,
                "date": dt,
                "monthly_return": compounded.loc[dt],
                "annualised_vol": ann_vol,
                "annualised_return": ann_ret,
                "sharpe_ratio": sharpe,
            })
    return pd.DataFrame(records)


def download_vix() -> pd.DataFrame:
    """Download VIX index (monthly average) as a market-risk proxy."""
    vix = yf.download("^VIX", start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix_close = vix.xs("Close", axis=1, level=0)
    else:
        vix_close = vix[["Close"]]
    vix_monthly = vix_close.resample("ME").mean()
    vix_monthly.columns = ["vix"]
    vix_monthly.index.name = "date"
    return vix_monthly


def build_panel_dataset() -> pd.DataFrame:
    """Merge financial metrics with ESG attributes into a panel dataset."""
    prices = download_stock_prices(TICKERS)
    prices.to_csv(os.path.join(DATA_DIR, "stock_prices.csv"))
    print(f"  → Saved daily prices ({prices.shape[0]} days × {prices.shape[1]} tickers)")

    monthly = compute_monthly_metrics(prices)
    if monthly.empty:
        raise RuntimeError("No monthly metrics computed — check network / tickers.")

    monthly["date"] = pd.to_datetime(monthly["date"])
    try:
        vix = download_vix()
        monthly = monthly.merge(vix.reset_index(), on="date", how="left")
    except Exception as exc:  # VIX is nice-to-have, not critical
        print(f"  ⚠ VIX download failed ({exc}) — continuing without it")
    if "vix" not in monthly.columns:
        monthly["vix"] = np.nan

    esg = get_esg_dataframe()
    panel = monthly.merge(esg, on="ticker", how="left")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

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
    print("Done.")
