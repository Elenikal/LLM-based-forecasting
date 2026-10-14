# Reading the Fed: live forecast record

Out-of-sample forecasts from the paper's models, issued and timestamped before
the outcomes are known. Nothing in `forecasts/` or `forecast_log.csv` is ever
edited after it is committed.

## Specification (unchanged from the paper)

- AR(4) vs AR(4) + 3 principal components of the 10 LLM tone scores
  (5 dimensions × FOMC minutes and Beige Book), 40-quarter rolling window,
  scores in levels, PCA re-fit on each window.
- **One change from the paper, declared before the first issue:** both models
  include dummies for 2020Q2, 2020Q3 and 2020Q4 (zero out of sample). Without
  them the three LLM PCs partly fit the COVID quarters and produce implausible
  GDP and unemployment calls. `backtest_covid.py` re-runs the 2020–2024
  backtest under this spec: AR+LLM beats AR in 9 of 12 (11 of 12 without).
- Targets: real GDP growth, CPI and core PCE inflation, unemployment,
  fed funds, 10Y–2Y spread. Horizons h = 1, 2 quarters.
- LLM scores 2007Q4–2024Q4 are the paper's (`outputs/llm_scores.csv`), frozen.
  Later documents use the same prompt, temperature 0.3 and 3 runs
  (`llm_scores_live.csv`, per-document detail in `doc_scores/`).

## What is different in real time

- **Ragged edge.** Each variable's window ends at its last complete quarter,
  so h = 1 is always the first quarter not yet published for that variable.
- **Outcomes are first releases** (ALFRED vintages). The paper used revised data.
- The model only uses text dated in or before the last in-window quarter.
  A Beige Book released in the current quarter enters at the next issue.

## Scorer

`claude-sonnet-4-5` (the paper's model), via `anthropic==0.86.0`
(`live/requirements.txt`). Anthropic retires this model on 2026-11-30. After that
the whole history is re-scored with one current model, the backtest is re-run on
those scores, and the switch date is logged here. Forecasts already issued stay
as they are.

## Each cycle (after the Beige Book, before the FOMC decision)

```bash
.venv-arm/bin/python live/forecast.py --dry-run   # look first
.venv-arm/bin/python live/forecast.py             # record it
git add live && git commit -m "Forecast YYYY-MM-DD" && git tag fc-YYYY-MM-DD
git push && git push --tags
.venv-arm/bin/python live/scorecard.py            # grade whatever has resolved
```

`forecasts/YYYY-MM-DD.md` holds the table for the post and the SHA-256 of the
JSON file. The tag pushed to GitHub is the timestamp of record.
