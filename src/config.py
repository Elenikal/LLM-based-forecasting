"""
config.py  —  Central configuration for the LLM-Augmented Forecasting pipeline.
All hyperparameters, paths, and constants live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env if present ───────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Project root ───────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CACHE_DIR  = ROOT / "cache"
TEXT_DIR   = CACHE_DIR / "texts"
SCORE_DIR  = CACHE_DIR / "scores"
OUTPUT_DIR = ROOT / "outputs"

for d in [TEXT_DIR / "fomc_minutes", TEXT_DIR / "beige_books", SCORE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API keys ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FRED_API_KEY      = os.environ.get("FRED_API_KEY", "")

# ── Sample & rolling-window design ────────────────────────────────────────────
EVAL_START   = "2020Q1"
EVAL_END     = "2024Q4"
ROLL_WINDOW  = 40
FORECAST_H   = [1, 2]

# ── BVAR (Minnesota prior) ────────────────────────────────────────────────────
BVAR_LAGS    = 4
BVAR_LAMBDA1 = 0.2
BVAR_LAMBDA2 = 0.5
BVAR_LAMBDA3 = 1.0

# Lenza-Primiceri (2022) COVID outlier treatment
# Observation variance scaled by V for COVID quarters → downweight by 1/sqrt(V)
OUTLIER_SCALE   = 100.0            # scale factor (Lenza-Primiceri use ~100)
COVID_OUTLIERS  = ["2020Q1", "2020Q2"]  # quarters treated as outliers
N_LLM_PCS    = 3

# ── LLM scoring ───────────────────────────────────────────────────────────────
LLM_MODEL      = "claude-sonnet-4-5"
LLM_TEMP       = 0.3
LLM_N_RUNS     = 3
LLM_MAX_TOKENS = 800

# ── FRED series ───────────────────────────────────────────────────────────────
BVAR_VARS = {
    "GDPC1":    "Real GDP (billions 2017$)",
    "CPIAUCSL": "CPI All Items",
    "PCEPILFE": "Core PCE Price Index",
    "UNRATE":   "Unemployment Rate",
    "FEDFUNDS": "Federal Funds Rate",
    "T10Y2Y":   "10Y-2Y Treasury Spread",
    "BAA10Y":   "BAA-10Y Credit Spread",
    "VIXCLS":   "CBOE VIX",
}

FORECAST_TARGETS = [
    "GDPC1_gr", "CPIAUCSL_yoy", "PCEPILFE_yoy",
    "UNRATE", "BAA10Y", "VIXCLS_log",
]

SCORE_DIMS = [
    "growth_expectations",
    "inflation_concern",
    "labor_market",
    "credit_conditions",
    "policy_uncertainty",
]

DM_TEST_LOSS = "squared"
SIGNIFICANCE = 0.10
RANDOM_SEED  = 42
