# LLM-Augmented Forecasting Pipeline — Run Guide

## Prerequisites

```bash
pip install fredapi anthropic pandas numpy scipy statsmodels scikit-learn \
            requests beautifulsoup4 tqdm python-dotenv pyarrow
```

Set environment variables (or create a `.env` file in this directory):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."        # https://console.anthropic.com
export FRED_API_KEY="your_key_here"          # free at https://fred.stlouisfed.org/docs/api/api_key.html
```

---

## First-time live run (step by step)

### Step 1 — Clear any stale demo cache

If you ran the pipeline in demo mode previously, delete the synthetic caches:

```bash
python reset_cache.py            # dry run — shows what will be deleted
python reset_cache.py --confirm  # actually deletes
```

This removes `cache/scores/score_matrix.parquet` (synthetic LLM scores) and
`cache/fred_data.parquet` (synthetic FRED data). Real scraped text files in
`cache/texts/` are preserved.

### Step 2 — Run the live pipeline

```bash
python src/pipeline.py --live
```

**What happens on first run:**

| Step | Time | What it does |
|------|------|--------------|
| FRED download | ~1 min | Downloads 8 FRED-QD series via API, caches to `cache/fred_data.parquet` |
| FOMC scraping | ~10 min | Scrapes ~200 minutes documents (2000–2024), caches as `.txt` files |
| Beige Book scraping | ~5 min | Scrapes ~200 Beige Books (2000–2024), caches as `.txt` files |
| LLM scoring | ~3–5 hrs | ~2,400 API calls × 3 runs = ~7,200 total; all cached — re-runs are instant |
| BVAR + evaluation | ~2 min | Rolling-window loop, DM tests, output files |

**Subsequent runs are instant** — all text and scores are cached to disk.

### Step 3 — Outputs

Results written to `outputs/`:

```
outputs/
  results_main.csv        RMSE, MAE, DM p-values for all models × targets × horizons
  results_subperiod.csv   Sub-period breakdown (COVID / inflation surge / soft landing)
  results_main.tex        LaTeX Table 3 (main results)
  llm_scores.csv          Full quarterly LLM score matrix (data appendix)
```

---

## URL notes (for reproducibility)

### FOMC Minutes

- Years 2000–2020: scraped from `federalreserve.gov/monetarypolicy/fomchistorical{year}.htm`
- Years 2021–2024: `fomchistorical{year}.htm` returns 404 (archive page not yet published).
  The pipeline falls back to `federalreserve.gov/monetarypolicy/fomccalendars.htm`
  which lists all recent meetings.

### Beige Books

- Scraped from `federalreserve.gov/releases/beigebook/{year}/`
- 8 issues per year; each links to a national summary page.
- **Do not use `minneapolisfed.org/beige-book-reports`** — that URL structure
  changed and the scraper will return 0 documents.

---

## Cost estimate (Anthropic API)

~2,400 documents × 3 runs × ~600 input tokens + ~300 output tokens ≈ **~5.4M tokens total**

At Sonnet pricing (~$3/M input, ~$15/M output):
- Input: ~4.3M tokens × $3 ≈ **$13**
- Output: ~2.2M tokens × $15 ≈ **$33**
- **Total: ~$2-3**

All calls are cached. You only pay once.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Results are synthetic / scores are random | `score_matrix.parquet` cached from demo run | `python reset_cache.py --confirm` |
| FOMC minutes: 0 docs for 2021–2024 | `fomchistorical{year}.htm` 404 | Fixed in v2 scraper — pull latest code |
| Beige Books: 0 docs | Wrong URL (minneapolisfed) | Fixed in v2 scraper — uses federalreserve.gov |
| API 403 / connection refused | Running in sandboxed environment | Must run locally; federalreserve.gov is not reachable from all cloud sandboxes |
| `FRED_API_KEY not set` | Missing env var | `export FRED_API_KEY=...` |
