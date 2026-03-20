"""
config.py
=========
Central configuration for the LLM-Augmented Forecasting pipeline.
All hyperparameters, paths, and constants live here.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CACHE_DIR   = ROOT / "cache"
TEXT_DIR    = CACHE_DIR / "texts"
SCORE_DIR   = CACHE_DIR / "scores"
OUTPUT_DIR  = ROOT / "outputs"

for d in [TEXT_DIR, SCORE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API keys (set as environment variables or .env file) ──────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FRED_API_KEY      = os.environ.get("FRED_API_KEY", "")          # free at fred.stlouisfed.org

# ── Sample & window design ────────────────────────────────────────────────────
TRAIN_START   = "2000Q1"    # start of full sample
EVAL_START    = "2020Q1"    # first out-of-sample forecast origin
EVAL_END      = "2024Q4"    # last out-of-sample forecast origin
ROLL_WINDOW   = 40          # quarters (10 years) for rolling estimation
FORECAST_H    = [1, 2]      # forecast horizons (quarters)

# ── BVAR hyperparameters (Minnesota prior) ────────────────────────────────────
BVAR_LAGS     = 4           # lag order p
BVAR_LAMBDA1  = 0.2         # overall tightness
BVAR_LAMBDA2  = 0.5         # cross-variable decay
BVAR_LAMBDA3  = 1.0         # lag decay exponent
N_LLM_PCS    = 3            # number of LLM PCs to retain

# ── LLM scoring ───────────────────────────────────────────────────────────────
LLM_MODEL      = "claude-haiku-4-5-20251001"  # ~20x cheaper than Sonnet
LLM_TEMP       = 0.3
LLM_N_RUNS     = 1          # single run — matches Haiku cost target
LLM_MAX_TOKENS = 600        # Haiku needs less headroom

# ── FRED-QD target variables (FRED mnemonics) ─────────────────────────────────
# These are the 8 variables in the BVAR system
BVAR_VARS = {
    "GDPC1":     "Real GDP (billions 2017$)",
    "CPIAUCSL":  "CPI All Items",
    "PCEPILFE":  "Core PCE Price Index",
    "UNRATE":    "Unemployment Rate",
    "FEDFUNDS":  "Federal Funds Rate",
    "T10Y2Y":    "10Y-2Y Treasury Spread",
    "BAA10Y":    "BAA-10Y Credit Spread",
    "VIXCLS":    "CBOE VIX",
}

# Forecast targets (subset of BVAR_VARS + transformations defined in data module)
FORECAST_TARGETS = ["GDPC1_gr", "CPIAUCSL_yoy", "PCEPILFE_yoy",
                    "UNRATE", "BAA10Y", "VIXCLS_log"]

# ── Text sources ──────────────────────────────────────────────────────────────
FOMC_BASE_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
BEIGE_BOOK_ARCHIVE = "https://www.minneapolisfed.org/beige-book-reports"

# ── Scoring dimensions ────────────────────────────────────────────────────────
SCORE_DIMS = [
    "growth_expectations",
    "inflation_concern",
    "labor_market",
    "credit_conditions",
    "policy_uncertainty",
]

# ── Evaluation ────────────────────────────────────────────────────────────────
DM_TEST_LOSS  = "squared"   # "squared" or "absolute"
SIGNIFICANCE  = 0.10        # p-value threshold for DM test reporting
RANDOM_SEED   = 42