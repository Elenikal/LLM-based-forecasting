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

## Headline set (declared 2026-10-14, before any outcome)

Posts headline inflation (CPI, core PCE), unemployment, the fed funds rate and
the 10Y–2Y spread: the variables where the corrected backtest and the COVID
robustness check find value in the text. GDP is still forecast, recorded in
`forecasts/` and graded by `scorecard.py`; it is not headlined, because the
text adds nothing for output once the COVID quarters are controlled for.

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

## Automatic updates

`.github/workflows/update-dashboard.yml` runs on GitHub at 11:00 and 18:00 ET on
weekdays (and on demand from the Actions tab). It grades recorded forecasts
against first-release data (`scorecard.py`), rebuilds `docs/data/site.json`, and
commits only if something changed. It never issues forecasts and aborts if
anything in `forecasts/`, `forecast_log.csv` or `benchmarks/` would change.
Needs the repository secret `FRED_API_KEY`. Because it commits to `main`,
run `git pull` before working locally.

## Each cycle (after the Beige Book, before the FOMC decision)

```bash
git pull                                          # pick up automatic updates
.venv-arm/bin/python live/forecast.py --dry-run   # look first
.venv-arm/bin/python live/forecast.py             # record it
git add live && git commit -m "Forecast YYYY-MM-DD" && git tag fc-YYYY-MM-DD
git push && git push --tags
.venv-arm/bin/python live/make_post.py YYYY-MM-DD    # table.png + post.md for Substack
.venv-arm/bin/python live/scorecard.py            # grade whatever has resolved
.venv-arm/bin/python live/build_site.py           # refresh docs/ (GitHub Pages dashboard)
```

`benchmarks/YYYY-MM-DD.csv` records the outside forecasts available on the
issue date (SPF, GDPNow, Cleveland Fed nowcast, fed funds futures) with their
definitions; `scorecard.py` grades them against the same first releases.

`forecasts/YYYY-MM-DD.md` holds the table for the post and the SHA-256 of the
JSON file. The tag pushed to GitHub is the timestamp of record.
