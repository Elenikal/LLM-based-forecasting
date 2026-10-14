#!/usr/bin/env python3
"""
live/forecast.py
================
Issue one dated, out-of-sample forecast from the Reading the Fed models.

Same specification as the paper (src/pipeline.py, Model 1 and Model 4):
  AR(4)        per variable, 40-quarter rolling window
  AR(4)+LLM    plus 3 PCs of the 10 LLM tone scores, PCA re-fit on the window,
               last in-window PC values held fixed over the horizon

Declared before the first issue (2026-10): both models add dummies for
2020Q2, 2020Q3 and 2020Q4, set to zero out of sample. Without them the three
LLM PCs partly fit the COVID quarters (see live/backtest_covid.py).

Live differences, all forced by real time:
  - each variable's window ends at its own last complete quarter (ragged edge),
    so h=1 is always the first quarter not yet published for that variable
  - LLM scores 2007Q4-2024Q4 are the frozen paper scores; later quarters are
    scored with the same prompt, temperature and number of runs

Usage
-----
  python live/forecast.py --dry-run     # everything, but writes nothing to the log
  python live/forecast.py               # issue and record the forecast
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import norm

LIVE = Path(__file__).resolve().parent
ROOT = LIVE.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(LIVE))

from config import BVAR_LAGS, N_LLM_PCS, ROLL_WINDOW, SCORE_DIR, SCORE_DIMS
from data.text_pull import group_by_quarter
from models.bvar import fit_ar, forecast_ar
from pipeline import rolling_pca
from backtest_covid import covid_dummies, stack
from sources import load_fred_live, load_new_texts

# Scorer for post-2024 documents. Must stay fixed for the life of the record.
LIVE_LLM_MODEL = "claude-sonnet-4-5"

FROZEN_SCORES = ROOT / "outputs" / "llm_scores.csv"      # paper, 2007Q4-2024Q4
LIVE_SCORES   = LIVE / "llm_scores_live.csv"
DOC_SCORES    = LIVE / "doc_scores"
FC_DIR        = LIVE / "forecasts"
LOG           = LIVE / "forecast_log.csv"

TARGETS = {
    "GDPC1_gr":     "Real GDP growth (q/q saar, %)",
    "CPIAUCSL_yoy": "CPI inflation (y/y, %)",
    "PCEPILFE_yoy": "Core PCE inflation (y/y, %)",
    "UNRATE":       "Unemployment rate (%)",
    "FEDFUNDS":     "Effective fed funds rate (%)",
    "T10Y2Y":       "10Y-2Y Treasury spread (pp)",
}
HORIZONS = [1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# LLM scores
# ─────────────────────────────────────────────────────────────────────────────

def update_scores(docs: list[dict]) -> pd.DataFrame:
    """Frozen paper scores + scores for every newer quarter. Paper values are
    never overwritten; a new source only fills a cell the paper left empty
    (Beige Book 2024)."""
    import anthropic
    from scoring.llm_scorer import build_score_matrix

    import scoring.llm_scorer as scorer
    scorer.LLM_MODEL = LIVE_LLM_MODEL

    frozen = pd.read_csv(FROZEN_SCORES, index_col=0)
    frozen.index = pd.PeriodIndex(frozen.index, freq="Q")

    last_frozen = frozen.index.max()
    todo = [d for d in docs
            if pd.Period(d["quarter"], freq="Q") > last_frozen
            or pd.isna(frozen.loc[d["quarter"], f"{d['source']}_{SCORE_DIMS[0]}"])]
    by_q = group_by_quarter(todo)
    new = build_score_matrix(by_q, anthropic.Anthropic())

    out = frozen.reindex(frozen.index.union(new.index))
    for q in new.index:
        for col in new.columns:
            if pd.isna(out.loc[q, col]) and not pd.isna(new.loc[q, col]):
                out.loc[q, col] = new.loc[q, col]

    # keep the per-document scores next to the record
    DOC_SCORES.mkdir(exist_ok=True)
    for q, by_src in by_q.items():
        for src, texts in by_src.items():
            if texts:
                for f in SCORE_DIR.glob(f"{src}_{q}_*.json"):
                    shutil.copy(f, DOC_SCORES / f.name)

    save = out.copy()
    save.index = save.index.astype(str)
    save.to_csv(LIVE_SCORES)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Forecasts
# ─────────────────────────────────────────────────────────────────────────────

def ar_sd(fit: dict, h: int) -> float:
    """h-step forecast s.d. from the AR lag polynomial (exog treated as known)."""
    phi = fit["b"][:fit["p"]]
    psi = [1.0]
    for j in range(1, h):
        psi.append(sum(phi[i] * psi[j - 1 - i] for i in range(min(j, len(phi)))))
    return float(np.sqrt(fit["sigma2"] * np.sum(np.square(psi))))


def make_forecasts(quant: pd.DataFrame, scores: pd.DataFrame) -> list[dict]:
    rows = []
    for var, label in TARGETS.items():
        s = quant[var].dropna()
        last_q = s.index[-1]
        win_idx = pd.period_range(last_q - (ROLL_WINDOW - 1), last_q, freq="Q")
        y = s.reindex(win_idx)
        if y.isna().any():
            raise ValueError(f"{var}: gaps inside the estimation window")
        if last_q not in scores.index or scores.loc[last_q].isna().all():
            raise ValueError(f"{var}: no LLM scores yet for {last_q}")

        pcs = rolling_pca(scores, win_idx, N_LLM_PCS)
        dum = covid_dummies(win_idx)
        H = max(HORIZONS)
        fit0 = fit_ar(y.values, p=BVAR_LAGS, exog=dum)
        fit1 = fit_ar(y.values, p=BVAR_LAGS, exog=stack(pcs, dum))
        k_d = 0 if dum is None else dum.shape[1]
        fc0 = forecast_ar(fit0, h=H, future_exog=np.zeros((H, k_d)) if k_d else None)
        fc1 = forecast_ar(fit1, h=H, future_exog=np.tile(
            np.hstack([pcs[-1:], np.zeros((1, k_d))]), (H, 1)))

        for h in HORIZONS:
            sd = ar_sd(fit1, h)
            rows.append({
                "variable": var, "label": label,
                "last_obs_quarter": str(last_q), "last_obs_value": round(float(s.iloc[-1]), 3),
                "h": h, "target_quarter": str(last_q + h),
                "AR": round(float(fc0[h - 1]), 3),
                "AR_LLM": round(float(fc1[h - 1]), 3),
                "AR_LLM_lo68": round(float(fc1[h - 1] - norm.ppf(.84) * sd), 3),
                "AR_LLM_hi68": round(float(fc1[h - 1] + norm.ppf(.84) * sd), 3),
                "AR_LLM_lo90": round(float(fc1[h - 1] - norm.ppf(.95) * sd), 3),
                "AR_LLM_hi90": round(float(fc1[h - 1] + norm.ppf(.95) * sd), 3),
            })
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def to_markdown(meta: dict, rows: list[dict]) -> str:
    lines = [f"Reading the Fed: forecast issued {meta['issued_utc'][:10]}", "",
             "| Variable | Last obs | Target | AR | AR+LLM | 68% band |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['last_obs_value']:.2f} ({r['last_obs_quarter']}) "
                     f"| {r['target_quarter']} | {r['AR']:.2f} | **{r['AR_LLM']:.2f}** "
                     f"| {r['AR_LLM_lo68']:.2f} to {r['AR_LLM_hi68']:.2f} |")
    lines += ["", f"Latest Fed text in the model: {meta['latest_documents']}",
              f"SHA-256 of the forecast file: {meta['sha256']}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, write nothing to forecasts/ or the log")
    args = ap.parse_args()

    issued = datetime.now(timezone.utc)
    print("── Text ──")
    docs = load_new_texts()
    print("── Scores ──")
    scores = update_scores(docs)
    print("── FRED ──")
    quant, last_obs = load_fred_live(issued.date())
    print("── Forecast ──")
    rows = make_forecasts(quant, scores)

    latest = {src: max((d["date"] for d in docs if d["source"] == src), default=None)
              for src in ["fomc_minutes", "beige_book"]}
    meta = {
        "issued_utc": issued.isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "llm_model_live": LIVE_LLM_MODEL,
        "anthropic_sdk": __import__("anthropic").__version__,
        "spec": f"AR({BVAR_LAGS}) vs AR({BVAR_LAGS})+{N_LLM_PCS} LLM PCs, "
                f"{ROLL_WINDOW}Q rolling window, levels, COVID dummies 2020Q2-Q4",
        "fred_last_observation": last_obs,
        "latest_documents": {"fomc_minutes_meeting": latest["fomc_minutes"],
                             "beige_book_release": latest["beige_book"]},
    }
    body = json.dumps({"meta": meta, "forecasts": rows}, indent=2)
    meta["sha256"] = hashlib.sha256(body.encode()).hexdigest()

    print(pd.DataFrame(rows)[["variable", "last_obs_quarter", "last_obs_value",
                              "target_quarter", "AR", "AR_LLM",
                              "AR_LLM_lo68", "AR_LLM_hi68"]].to_string(index=False))

    if args.dry_run:
        print("\n(dry run: nothing recorded)")
        return

    FC_DIR.mkdir(exist_ok=True)
    stem = FC_DIR / issued.strftime("%Y-%m-%d")
    if stem.with_suffix(".json").exists():
        sys.exit(f"{stem}.json already exists; a forecast is never overwritten.")
    stem.with_suffix(".json").write_text(body)
    stem.with_suffix(".md").write_text(to_markdown(meta, rows))

    log = pd.DataFrame(rows).assign(issued_utc=meta["issued_utc"], sha256=meta["sha256"])
    log.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print(f"\nRecorded {stem.name}  sha256={meta['sha256']}")
    print("Next: git add live && git commit && git tag fc-" + stem.name + " && git push --tags")


if __name__ == "__main__":
    main()
