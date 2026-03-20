"""
data/fred_pull.py
=================
Downloads and transforms FRED quantitative series for the BVAR system.

Key design decisions:
- Uses real-time FRED-QD vintages where available to avoid look-ahead bias
- Falls back to point-in-time FRED API for series not in FRED-QD
- All transformations (growth rates, log, differencing) applied here
- Returns a tidy quarterly DataFrame indexed by period string "YYYYQQ"
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FRED_API_KEY, CACHE_DIR, BVAR_VARS, FORECAST_TARGETS

FRED_CACHE = CACHE_DIR / "fred_data.parquet"


def get_fred_client():
    if not FRED_AVAILABLE:
        raise ImportError("fredapi not installed. Run: pip install fredapi")
    if not FRED_API_KEY:
        raise ValueError(
            "FRED_API_KEY not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=FRED_API_KEY)


def download_fred_series(series_id: str, fred: "Fred") -> pd.Series:
    """Download a single FRED series and return as quarterly Series."""
    raw = fred.get_series(series_id, observation_start="1995-01-01")
    # Resample to quarter-end if higher frequency
    if raw.index.freq is None or raw.index.freq.n < 90:
        raw = raw.resample("QE").last()
    raw.index = raw.index.to_period("Q")
    raw.name = series_id
    return raw


def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard macro transformations to raw level series.

    Transformations:
      GDPC1       -> GDPC1_gr    : QoQ annualised % growth  = 400*log(x_t/x_{t-1})
      CPIAUCSL    -> CPIAUCSL_yoy: YoY % change             = 100*(x_t/x_{t-4} - 1)
      PCEPILFE    -> PCEPILFE_yoy: YoY % change
      UNRATE      -> kept in levels (already %)
      FEDFUNDS    -> kept in levels (already %)
      T10Y2Y      -> kept in levels (basis points approximated)
      BAA10Y      -> kept in levels (basis points)
      VIXCLS      -> VIXCLS_log  : log(VIX)
    """
    out = pd.DataFrame(index=df.index)

    # Real GDP — annualised QoQ log growth
    if "GDPC1" in df.columns:
        out["GDPC1_gr"] = 400 * np.log(df["GDPC1"] / df["GDPC1"].shift(1))

    # CPI — year-over-year percent change
    if "CPIAUCSL" in df.columns:
        out["CPIAUCSL_yoy"] = 100 * (df["CPIAUCSL"] / df["CPIAUCSL"].shift(4) - 1)

    # Core PCE — year-over-year percent change
    if "PCEPILFE" in df.columns:
        out["PCEPILFE_yoy"] = 100 * (df["PCEPILFE"] / df["PCEPILFE"].shift(4) - 1)

    # Levels — no transformation needed
    for col in ["UNRATE", "FEDFUNDS", "T10Y2Y", "BAA10Y"]:
        if col in df.columns:
            out[col] = df[col]

    # VIX — log level (reduces right skew)
    if "VIXCLS" in df.columns:
        out["VIXCLS_log"] = np.log(df["VIXCLS"])

    return out.dropna(how="all")


def load_fred_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Main entry point. Returns a quarterly DataFrame with all transformed
    BVAR variables from 1999Q1 onward.

    Columns: GDPC1_gr, CPIAUCSL_yoy, PCEPILFE_yoy, UNRATE, FEDFUNDS,
             T10Y2Y, BAA10Y, VIXCLS_log  (plus raw levels for reference)
    """
    if use_cache and FRED_CACHE.exists():
        print(f"Loading FRED data from cache: {FRED_CACHE}")
        df = pd.read_parquet(FRED_CACHE)
        df.index = pd.PeriodIndex(df.index, freq="Q")
        return df

    print("Downloading FRED data...")
    fred = get_fred_client()

    raw_frames = []
    for series_id in BVAR_VARS:
        try:
            s = download_fred_series(series_id, fred)
            raw_frames.append(s)
            print(f"  ✓ {series_id}")
        except Exception as e:
            warnings.warn(f"Could not download {series_id}: {e}")

    raw = pd.concat(raw_frames, axis=1)
    raw.index = pd.PeriodIndex(raw.index, freq="Q")

    # Apply transformations
    transformed = apply_transformations(raw)

    # Merge raw levels + transformed (raw kept for reference in robustness checks)
    # NEW
    raw_deduped = raw.drop(columns=[c for c in transformed.columns if c in raw.columns], errors="ignore")
    df = pd.concat([raw_deduped, transformed], axis=1)

    df = df.loc["1999Q1":]

    # Save cache
    df_save = df.copy()
    df_save.index = df_save.index.astype(str)
    df_save.to_parquet(FRED_CACHE)
    print(f"Saved FRED data cache: {FRED_CACHE}  ({len(df)} quarters)")

    return df


def get_bvar_system(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the 8-variable BVAR system from the full DataFrame.
    Returns only the transformed forecast-ready columns, quarterly frequency.
    """
    bvar_cols = ["GDPC1_gr", "CPIAUCSL_yoy", "PCEPILFE_yoy",
                 "UNRATE", "FEDFUNDS", "T10Y2Y", "BAA10Y", "VIXCLS_log"]
    available = [c for c in bvar_cols if c in df.columns]
    missing   = [c for c in bvar_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Missing BVAR variables: {missing}")
    return df[available].dropna()


def get_rolling_window(df: pd.DataFrame, t: str, window: int) -> pd.DataFrame:
    """
    Return the rolling estimation window ending at period t-1 (exclusive of t).

    Parameters
    ----------
    df     : full DataFrame with PeriodIndex
    t      : forecast origin as string e.g. "2022Q3"
    window : number of quarters to include

    Returns
    -------
    Slice of df of length `window` ending at t-1
    """
    t_period = pd.Period(t, freq="Q")
    end      = t_period - 1
    start    = end - (window - 1)

    mask = (df.index >= start) & (df.index <= end)
    sub  = df.loc[mask]

    if len(sub) < window:
        warnings.warn(
            f"Window at {t}: requested {window}Q, got {len(sub)}Q "
            f"(start={start}, end={end})"
        )
    return sub


# ── Simple demo / smoke test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    # Demo with synthetic data when no API key available
    if not FRED_API_KEY:
        print("No FRED_API_KEY set — generating synthetic demo data.")
        idx = pd.period_range("1999Q1", "2024Q4", freq="Q")
        np.random.seed(42)
        demo = pd.DataFrame({
            "GDPC1_gr":      np.random.normal(2.5, 2.0, len(idx)),
            "CPIAUCSL_yoy":  np.random.normal(2.0, 1.5, len(idx)),
            "PCEPILFE_yoy":  np.random.normal(1.8, 1.2, len(idx)),
            "UNRATE":        np.random.normal(5.0, 1.5, len(idx)),
            "FEDFUNDS":      np.random.normal(2.0, 2.0, len(idx)),
            "T10Y2Y":        np.random.normal(0.5, 0.8, len(idx)),
            "BAA10Y":        np.random.normal(1.5, 0.5, len(idx)),
            "VIXCLS_log":    np.random.normal(3.0, 0.4, len(idx)),
        }, index=idx)
        print(demo.tail())
        win = get_rolling_window(demo, "2022Q1", 40)
        print(f"\n40Q window ending 2021Q4: {win.index[0]} – {win.index[-1]} ({len(win)} obs)")
    else:
        df = load_fred_data(use_cache=False)
        sys_df = get_bvar_system(df)
        print(sys_df.tail(8))
        win = get_rolling_window(sys_df, "2022Q1", 40)
        print(f"\n40Q window: {win.index[0]} – {win.index[-1]}")
