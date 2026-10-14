#!/usr/bin/env python3
"""
live/scorecard.py
=================
Grade every recorded forecast whose target quarter has been published.

Outcomes are FIRST-RELEASE values (ALFRED vintages), i.e. what was known the
day the number came out, not today's revised history. The paper's backtest
used revised data; the live record does not.

  python live/scorecard.py      → live/scorecard.csv + summary
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

LIVE = Path(__file__).resolve().parent
load_dotenv(LIVE.parent / ".env")
sys.path.insert(0, str(LIVE.parent / "src"))
from config import FRED_API_KEY

LOG = LIVE / "forecast_log.csv"
OUT = LIVE / "scorecard.csv"

# variable → (FRED id, transformation)
SERIES = {
    "GDPC1_gr":     ("GDPC1",    "qoq_saar"),
    "CPIAUCSL_yoy": ("CPIAUCSL", "yoy"),
    "PCEPILFE_yoy": ("PCEPILFE", "yoy"),
    "UNRATE":       ("UNRATE",   "level"),
    "FEDFUNDS":     ("FEDFUNDS", "level"),
    "T10Y2Y":       ("T10Y2Y",   "level_daily"),
}


@lru_cache(maxsize=None)
def _fred():
    from fredapi import Fred
    return Fred(api_key=FRED_API_KEY)


@lru_cache(maxsize=None)
def _releases(sid: str) -> pd.DataFrame:
    df = _fred().get_series_all_releases(sid)
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    return df


def _vintage(sid: str, obs_date: pd.Timestamp) -> pd.Series | None:
    """The series as published on the day obs_date first appeared."""
    rel = _releases(sid)
    first = rel.loc[rel["date"] == obs_date, "realtime_start"]
    if first.empty:
        return None
    v = rel[rel["realtime_start"] <= first.min()]
    v = v.sort_values("realtime_start").groupby("date")["value"].last()
    return pd.to_numeric(v, errors="coerce")


def first_release(var: str, quarter: str) -> float | None:
    sid, how = SERIES[var]
    q = pd.Period(quarter, freq="Q")
    if how == "level_daily":                  # market data, not revised
        if pd.Timestamp.today() <= q.end_time:
            return None
        s = _fred().get_series(sid, q.start_time, q.end_time).dropna()
        return float(s.iloc[-1]) if len(s) else None
    if how == "qoq_saar":
        d = q.start_time.normalize()
        v = _vintage(sid, d)
        if v is None:
            return None
        return float(400 * np.log(v[d] / v[(q - 1).start_time.normalize()]))
    m = q.asfreq("M", how="end").start_time.normalize()   # last month of quarter
    v = _vintage(sid, m)
    if v is None:
        return None
    if how == "yoy":
        return float(100 * (v[m] / v[m - pd.DateOffset(years=1)] - 1))
    return float(v[m])


def main():
    if not LOG.exists():
        sys.exit("No forecasts recorded yet.")
    log = pd.read_csv(LOG)
    log["actual"] = [first_release(v, q) for v, q in
                     zip(log["variable"], log["target_quarter"])]
    done = log.dropna(subset=["actual"]).copy()
    done["err_AR"] = done["AR"] - done["actual"]
    done["err_AR_LLM"] = done["AR_LLM"] - done["actual"]
    done["LLM_closer"] = done["err_AR_LLM"].abs() < done["err_AR"].abs()
    done["in_68"] = done["actual"].between(done["AR_LLM_lo68"], done["AR_LLM_hi68"])
    done["in_90"] = done["actual"].between(done["AR_LLM_lo90"], done["AR_LLM_hi90"])
    done["direction_right"] = (np.sign(done["AR_LLM"] - done["last_obs_value"]) ==
                               np.sign(done["actual"] - done["last_obs_value"]))
    done.to_csv(OUT, index=False)

    print(f"{len(done)} of {len(log)} forecasts resolved\n")
    if done.empty:
        return
    stats = dict(
        n=("actual", "size"),
        rmse_AR=("err_AR", lambda e: np.sqrt(np.mean(e ** 2))),
        rmse_AR_LLM=("err_AR_LLM", lambda e: np.sqrt(np.mean(e ** 2))),
        LLM_win_rate=("LLM_closer", "mean"),
        cover_68=("in_68", "mean"),
        cover_90=("in_90", "mean"),
        direction=("direction_right", "mean"),
    )
    # RMSEs are per variable (different units); hit rates also pooled
    summ = done.groupby(["variable", "h"]).agg(**stats)
    summ["rmse_ratio"] = summ["rmse_AR_LLM"] / summ["rmse_AR"]
    print(summ.round(3).to_string())
    pooled = done[["LLM_closer", "in_68", "in_90", "direction_right"]].mean()
    print(f"\nAll resolved forecasts (n={len(done)}): "
          f"AR+LLM closer than AR {pooled['LLM_closer']:.0%} | "
          f"inside 68% band {pooled['in_68']:.0%} | inside 90% band {pooled['in_90']:.0%} | "
          f"direction right {pooled['direction_right']:.0%}")


if __name__ == "__main__":
    main()
