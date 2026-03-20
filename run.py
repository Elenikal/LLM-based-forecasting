#!/usr/bin/env python3
"""
run.py
======
Entry point for the LLM-Augmented Forecasting pipeline.

Usage
-----
  # Demo mode — no API keys needed, uses synthetic data (~5 seconds)
  python run.py --demo

  # Live mode — requires .env with ANTHROPIC_API_KEY and FRED_API_KEY
  python run.py --live

  # Live mode, skip re-downloading already-cached data
  python run.py --live --use-cache

Output files written to ./outputs/
  results_main.csv        RMSE, MAE, DM stats for all models
  results_subperiod.csv   Sub-period breakdown (COVID / inflation / soft landing)
  llm_scores.csv          Quarterly LLM score vectors  (data appendix)
  results_main.tex        LaTeX Table 3 (ready to paste into paper)
"""

import sys
import argparse
from pathlib import Path

# Make sure src/ is on the path regardless of where run.py is called from
SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="LLM-Augmented Forecasting of U.S. GDP and Financial Indicators"
)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--demo", action="store_true",
                  help="Synthetic data — no API keys needed. Use to verify installation.")
mode.add_argument("--live", action="store_true",
                  help="Real data — requires ANTHROPIC_API_KEY and FRED_API_KEY in .env")
parser.add_argument("--use-cache", action="store_true", default=True,
                    help="Use cached FRED data and LLM scores if available (default: True)")
parser.add_argument("--no-cache", dest="use_cache", action="store_false",
                    help="Force re-download of all data and re-score all documents")
args = parser.parse_args()

# ── Key checks before doing anything expensive ───────────────────────────────
if args.live:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    import os
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("FRED_API_KEY"):
        missing.append("FRED_API_KEY")
    if missing:
        print("\n  ✗  Missing API keys:", ", ".join(missing))
        print("     Copy .env.example → .env and fill in your keys.")
        print("     Free FRED key:      https://fred.stlouisfed.org/docs/api/api_key.html")
        print("     Anthropic key:      https://console.anthropic.com\n")
        sys.exit(1)

    # Preflight: verify Anthropic API key + model work before the full run
    print("  Checking Anthropic API... ", end="", flush=True)
    try:
        import anthropic as _ant
        from src.config import LLM_MODEL
        _client = _ant.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        _resp = _client.messages.create(
            model=LLM_MODEL, max_tokens=10,
            messages=[{"role": "user", "content": "Reply with the word OK."}],
        )
        print(f"✓  (model: {LLM_MODEL})")
    except Exception as _e:
        print(f"\n  ✗  Anthropic API check failed: {_e}")
        print(f"     Model used: {LLM_MODEL}")
        print("     Check your ANTHROPIC_API_KEY and internet connection.\n")
        sys.exit(1)

# ── Import and run ────────────────────────────────────────────────────────────
from pipeline import main

main(demo=args.demo, use_cache=args.use_cache)
