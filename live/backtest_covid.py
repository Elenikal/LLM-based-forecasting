#!/usr/bin/env python3
"""
live/backtest_covid.py
======================
Re-run the paper's AR vs AR+LLM backtest (2020Q1-2024Q4 origins, cached FRED
data and cached LLM scores, no API calls) under:

  paper   : exactly as src/pipeline.py
  covid   : + dummies for 2020Q2, 2020Q3, 2020Q4 in both AR and AR+LLM
            (dummy set to 0 out of sample; dropped when not in the window)

and scores each with two target alignments:

  as_paper : every horizon compared with the value at the origin quarter t
             (what evaluate_and_export does)
  aligned  : the h-step forecast compared with the value at t+h-1

  python live/backtest_covid.py   → live/backtest_covid.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = Path(__file__).resolve().parent
ROOT = LIVE.parent
sys.path.insert(0, str(ROOT / "src"))

from config import BVAR_LAGS, EVAL_END, EVAL_START, N_LLM_PCS, ROLL_WINDOW, SCORE_DIR
from data.fred_pull import get_bvar_system
from evaluation.metrics import diebold_mariano
from models.bvar import fit_ar, forecast_ar
from pipeline import rolling_pca

TARGETS = ["GDPC1_gr", "CPIAUCSL_yoy", "PCEPILFE_yoy", "UNRATE", "FEDFUNDS", "T10Y2Y"]
COVID = ["2020Q2", "2020Q3", "2020Q4"]
H = [1, 2]


def covid_dummies(win_idx: pd.PeriodIndex) -> np.ndarray | None:
    cols = [[float(str(q) == c) for q in win_idx] for c in COVID]
    cols = [c for c in cols if any(c)]
    return np.array(cols).T if cols else None


def stack(*blocks):
    blocks = [b for b in blocks if b is not None]
    return np.hstack(blocks) if blocks else None


def run(quant: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in pd.period_range(EVAL_START, EVAL_END, freq="Q"):
        win_idx = pd.period_range(t - ROLL_WINDOW, t - 1, freq="Q")
        Y = quant.loc[quant.index.isin(win_idx)].dropna()       # as pipeline.py
        pcs = rolling_pca(scores, Y.index, N_LLM_PCS)
        dum = covid_dummies(Y.index)
        for v in TARGETS:
            y = Y[v].values
            specs = {
                ("paper", "AR"):      (None, None),
                ("paper", "AR_LLM"):  (pcs, pcs[-1:]),
                ("covid", "AR"):      (dum, None),
                ("covid", "AR_LLM"):  (stack(pcs, dum), pcs[-1:]),
            }
            for (spec, model), (X, x_last) in specs.items():
                fit = fit_ar(y, p=BVAR_LAGS, exog=X)
                fut = None
                if X is not None:
                    k_d = X.shape[1] - (x_last.shape[1] if x_last is not None else 0)
                    last = x_last if x_last is not None else np.zeros((1, 0))
                    fut = np.tile(np.hstack([last, np.zeros((1, k_d))]), (max(H), 1))
                fc = forecast_ar(fit, h=max(H), future_exog=fut)
                for h in H:
                    rows.append({"origin": str(t), "variable": v, "spec": spec,
                                 "model": model, "h": h, "forecast": fc[h - 1],
                                 "actual_as_paper": quant.loc[t, v],
                                 "actual_aligned": quant.loc[t + h - 1, v]})
    return pd.DataFrame(rows)


def summarise(fc: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (spec, v, h), g in fc.groupby(["spec", "variable", "h"]):
        ar = g[g.model == "AR"].set_index("origin")
        ll = g[g.model == "AR_LLM"].set_index("origin")
        for align in ["as_paper", "aligned"]:
            a = ar[f"actual_{align}"].values
            r_ar = np.sqrt(np.mean((ar.forecast.values - a) ** 2))
            r_ll = np.sqrt(np.mean((ll.forecast.values - a) ** 2))
            dm = diebold_mariano(a, ll.forecast.values, ar.forecast.values, h=h)
            out.append({"spec": spec, "alignment": align, "variable": v, "h": h,
                        "rmse_AR": r_ar, "rmse_AR_LLM": r_ll,
                        "ratio": r_ll / r_ar, "dm_pval": dm.get("p_value", np.nan)})
    return pd.DataFrame(out)


def main():
    quant = pd.read_parquet(ROOT / "cache" / "fred_data.parquet")
    quant.index = pd.PeriodIndex(quant.index, freq="Q")
    quant = get_bvar_system(quant)
    scores = pd.read_parquet(SCORE_DIR / "score_matrix.parquet")
    scores.index = pd.PeriodIndex(scores.index, freq="Q")

    fc = run(quant, scores)
    res = summarise(fc)
    res.to_csv(LIVE / "backtest_covid.csv", index=False)

    wide = res.pivot_table(index=["variable", "h"], columns=["spec", "alignment"],
                           values="ratio")
    print("AR+LLM / AR RMSE ratio (<1 = LLM helps)\n")
    print(wide.round(3).to_string())
    print("\nWins out of 12:")
    print((wide < 1).sum().to_string())


if __name__ == "__main__":
    main()
