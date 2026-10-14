#!/usr/bin/env python3
"""
live/build_site.py
==================
Pack the backtest results and the live record into docs/data/site.json for the
GitHub Pages dashboard (docs/index.html). Run after forecast.py / scorecard.py.

  python live/build_site.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

LIVE = Path(__file__).resolve().parent
ROOT = LIVE.parent
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(LIVE))
sys.path.insert(0, str(ROOT / "src"))

from make_post import HEADLINE
from forecast import TARGETS

KEEP = list(TARGETS)
REPO = "https://github.com/Elenikal/LLM-based-forecasting"


def records(df: pd.DataFrame) -> list[dict]:
    df = df.replace({np.nan: None})
    return json.loads(df.to_json(orient="records"))


def results(path: Path) -> list[dict]:
    d = pd.read_csv(path)
    d = d[d.variable.isin(KEEP)]
    cols = ["variable", "horizon", "AR_rmse", "BVAR_ratio", "BVAR_LLM_ratio",
            "AR_LLM_ratio", "AR_LLM_dm_pval", "AR_LLM_dm_sig", "BVAR_LLM_dm_sig"]
    return records(d[cols].fillna({"AR_LLM_dm_sig": "", "BVAR_LLM_dm_sig": ""}))


def main():
    site = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo": REPO, "headline": HEADLINE, "labels": TARGETS}

    # Backtest (paper, corrected)
    site["main"] = results(OUT / "results_main.csv")
    site["main_covid"] = results(OUT / "results_main_covid.csv")
    sub = pd.read_csv(OUT / "results_subperiod.csv")
    site["subperiod"] = records(sub[sub.variable.isin(KEEP)])

    # LLM scores: paper sample + live quarters
    sc = pd.read_csv(LIVE / "llm_scores_live.csv", index_col=0)
    site["scores"] = {"quarters": list(sc.index),
                      **{c: [None if pd.isna(v) else round(float(v), 3) for v in sc[c]]
                         for c in sc.columns}}

    # Live record
    issues = []
    for f in sorted((LIVE / "forecasts").glob("*.json")):
        raw = f.read_bytes()
        rec = json.loads(raw)
        rec["meta"]["sha256"] = hashlib.sha256(raw).hexdigest()
        bf = LIVE / "benchmarks" / f"{f.stem}.csv"
        bench = records(pd.read_csv(bf)) if bf.exists() else []
        issues.append({"issued": f.stem, "tag": f"fc-{f.stem}",
                       "meta": rec["meta"], "forecasts": rec["forecasts"],
                       "benchmarks": bench})
    site["issues"] = issues
    sc_file = LIVE / "scorecard.csv"
    site["scorecard"] = records(pd.read_csv(sc_file)) if sc_file.exists() and \
        sc_file.stat().st_size > 1 else []

    # Recent history for the live charts (current vintage)
    try:
        from sources import load_fred_live
        q, _ = load_fred_live()
        q = q.loc["2024Q1":, KEEP]
        site["history"] = {"quarters": [str(i) for i in q.index],
                           **{c: [None if pd.isna(v) else round(float(v), 3) for v in q[c]]
                              for c in KEEP}}
    except Exception as e:                       # no FRED key: charts show forecasts only
        print(f"  (history skipped: {e})")
        site["history"] = None

    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "data" / "site.json").write_text(json.dumps(site, indent=1))
    (DOCS / ".nojekyll").touch()
    print(f"Wrote {DOCS / 'data' / 'site.json'}")


if __name__ == "__main__":
    main()
