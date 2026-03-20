"""
scoring/llm_scorer.py
=====================
Calls the Claude API to extract structured 5-dimensional scores from
central bank and earnings call documents.

Key design choices:
- LLM is NEVER asked to forecast; it scores what it reads
- "treat document as undated" instruction prevents memorisation bias
- Each document scored LLM_N_RUNS times; mean taken, SD reported
- Results cached to disk; re-running is idempotent
- JSON output validated against schema before acceptance
"""

import json
import hashlib
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import anthropic

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMP, LLM_N_RUNS,
    LLM_MAX_TOKENS, SCORE_DIMS, SCORE_DIR
)

# ── Prompt components ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a monetary policy analyst at a central bank research department.
Your task is to score a central bank or corporate document on five dimensions \
of macroeconomic tone using the text provided.

Instructions:
1. Score each dimension on a scale from -2 to +2 in 0.5-point increments.
   (-2 = strongly negative / contractionary / concerned)
   (+2 = strongly positive / expansionary / confident)
2. Write a 1–2 sentence rationale for each score quoting brief phrases from the text.
3. Return ONLY a valid JSON object with the exact schema shown — no preamble, \
no markdown fencing, no extra keys.
4. Do NOT use information from outside the document. \
Treat the document as undated and anonymous; do not infer time period or author.

Scoring dimension definitions:
- growth_expectations : Tone about current and near-term real economic output and activity
- inflation_concern   : Degree of worry or alarm expressed about price pressures
- labor_market        : Assessment of employment conditions (tight=+2, slack=-2)
- credit_conditions   : Ease of financial/credit conditions (easy=+2, tight=-2)
- policy_uncertainty  : Level of expressed uncertainty about the forward policy path \
(high uncertainty=+2, clear path=-2)
"""

USER_TEMPLATE = """\
Analyze the following document and return a JSON object with this exact schema:

{{
  "growth_expectations": {{"score": <float -2 to 2 in 0.5 steps>, "rationale": "<10 words max>"}},
  "inflation_concern":   {{"score": <float -2 to 2 in 0.5 steps>, "rationale": "<10 words max>"}},
  "labor_market":        {{"score": <float -2 to 2 in 0.5 steps>, "rationale": "<10 words max>"}},
  "credit_conditions":   {{"score": <float -2 to 2 in 0.5 steps>, "rationale": "<10 words max>"}},
  "policy_uncertainty":  {{"score": <float -2 to 2 in 0.5 steps>, "rationale": "<10 words max>"}}
}}

Document text:
--------------
{document}
--------------
Return only the JSON object.
"""

# ── Document hashing for cache keys ──────────────────────────────────────────

def doc_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ── Schema validation ─────────────────────────────────────────────────────────

VALID_SCORES = {x / 2 for x in range(-4, 5)}   # {-2, -1.5, ..., 1.5, 2}

def validate_response(data: dict) -> bool:
    """Check that the API response matches the expected schema."""
    if not isinstance(data, dict):
        return False
    for dim in SCORE_DIMS:
        if dim not in data:
            return False
        entry = data[dim]
        if not isinstance(entry, dict):
            return False
        if "score" not in entry or "rationale" not in entry:
            return False
        score = float(entry["score"])
        if score not in VALID_SCORES:
            # Round to nearest 0.5 and warn
            rounded = round(score * 2) / 2
            rounded = max(-2.0, min(2.0, rounded))
            entry["score"] = rounded
            warnings.warn(f"Score {score} rounded to {rounded}")
    return True


# ── Single-document scorer ────────────────────────────────────────────────────

def score_document_once(text: str, client: anthropic.Anthropic) -> Optional[dict]:
    """
    Call Claude API once for a single document.
    Returns parsed JSON dict or None on failure.
    """
    # Truncate very long documents (FOMC minutes ~15k words; keep first 6k words)
    words = text.split()
    if len(words) > 6000:
        text = " ".join(words[:6000]) + "\n[... document truncated for length ...]"

    prompt = USER_TEMPLATE.format(document=text)

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMP,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fencing if present
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        data = json.loads(raw)
        if validate_response(data):
            return data
        else:
            warnings.warn("API response failed schema validation.")
            return None

    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  ✗ API error ({type(e).__name__}): {e}")
        time.sleep(5)
        return None


def score_document(text: str, client: anthropic.Anthropic,
                   n_runs: int = LLM_N_RUNS,
                   cache_key: Optional[str] = None) -> Optional[dict]:
    """
    Score a document n_runs times and return aggregated result.

    Returns dict with:
      {dim: {"mean": float, "std": float, "runs": [float, ...]}, ...}
      plus "reliability_ok": bool  (True if max SD across dims < 0.5)
    """
    # Check cache
    if cache_key:
        cache_path = SCORE_DIR / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    run_results = []
    for i in range(n_runs):
        result = score_document_once(text, client)
        if result is not None:
            run_results.append(result)
        time.sleep(0.5)

    if not run_results:
        return None

    # Aggregate across runs
    aggregated = {}
    max_sd = 0.0
    for dim in SCORE_DIMS:
        scores_raw = [r[dim]["score"] for r in run_results if dim in r]
        rationale  = run_results[0][dim].get("rationale", "")  # use first run's rationale
        if not scores_raw:
            continue
        mean_s = float(np.mean(scores_raw))
        std_s  = float(np.std(scores_raw)) if len(scores_raw) > 1 else 0.0
        max_sd = max(max_sd, std_s)
        aggregated[dim] = {
            "mean":      mean_s,
            "std":       std_s,
            "runs":      scores_raw,
            "rationale": rationale,
        }

    aggregated["reliability_ok"] = max_sd < 0.5
    aggregated["n_runs_succeeded"] = len(run_results)

    # Save to cache
    if cache_key:
        cache_path = SCORE_DIR / f"{cache_key}.json"
        cache_path.write_text(json.dumps(aggregated, indent=2))

    return aggregated


# ── Quarter-level scorer ──────────────────────────────────────────────────────

def score_quarter(quarter: str, texts_by_source: dict,
                  client: anthropic.Anthropic) -> dict:
    """
    Score all documents for a given quarter.

    texts_by_source: {"fomc_minutes": [text, ...], "beige_book": [text, ...]}

    Returns:
    {
      "quarter": str,
      "fomc_minutes":  5-dim mean vector (or None),
      "beige_book":    5-dim mean vector (or None),
      "n_docs": int,
    }
    """
    source_vectors = {}

    for source, doc_list in texts_by_source.items():
        if not doc_list:
            source_vectors[source] = None
            continue

        # Score each document in this source
        dim_scores = {dim: [] for dim in SCORE_DIMS}
        for i, text in enumerate(doc_list):
            key = f"{source}_{quarter}_{i}_{doc_hash(text)}"
            result = score_document(text, client, cache_key=key)
            if result:
                for dim in SCORE_DIMS:
                    if dim in result:
                        dim_scores[dim].append(result[dim]["mean"])

        # Average across documents in this source
        if any(dim_scores[dim] for dim in SCORE_DIMS):
            source_vectors[source] = {
                dim: float(np.mean(scores)) if scores else 0.0
                for dim, scores in dim_scores.items()
            }
        else:
            source_vectors[source] = None

    return {
        "quarter":      quarter,
        "fomc_minutes": source_vectors.get("fomc_minutes"),
        "beige_book":   source_vectors.get("beige_book"),
        "n_docs":       sum(len(v) for v in texts_by_source.values()),
    }


# ── Full corpus scorer ────────────────────────────────────────────────────────

def build_score_matrix(texts: dict, client: anthropic.Anthropic,
                       quarters: Optional[list] = None) -> pd.DataFrame:
    """
    Score all quarters in `texts` and return a DataFrame indexed by quarter.

    Columns: one per (source × dimension), e.g. "fomc_minutes_growth_expectations"
    Shape: (n_quarters, n_sources × n_dims)

    Parameters
    ----------
    texts    : output of text_pull.load_all_texts()
    client   : Anthropic client
    quarters : optional list of quarter strings to process (default: all)
    """
    if quarters is None:
        quarters = sorted(texts.keys())

    all_rows = []
    for q in quarters:
        if q not in texts:
            continue
        print(f"Scoring {q}...", end=" ", flush=True)
        result = score_quarter(q, texts[q], client)
        print(f"({result['n_docs']} docs)")

        row = {"quarter": q}
        for source in ["fomc_minutes", "beige_book"]:
            vec = result.get(source)
            for dim in SCORE_DIMS:
                col = f"{source}_{dim}"
                row[col] = vec[dim] if vec else np.nan

        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.index = pd.PeriodIndex(df["quarter"], freq="Q")
    df = df.drop(columns=["quarter"])
    return df


def get_llm_features(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the 10-column score matrix (5 dims × 2 sources: FOMC + Beige).
    Forward-fill missing quarters (at most 1Q), then drop remaining NaNs.
    """
    feature_cols = [
        f"{src}_{dim}"
        for src in ["fomc_minutes", "beige_book"]
        for dim in SCORE_DIMS
    ]
    available = [c for c in feature_cols if c in score_matrix.columns]
    feats = score_matrix[available].copy()
    feats = feats.ffill(limit=1)   # fill at most 1 missing quarter
    return feats.dropna(how="all")


# ── Demo / smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY not set — running mock demo.\n")

        # Mock client for offline testing
        class MockClient:
            class messages:
                @staticmethod
                def create(**kwargs):
                    mock_json = json.dumps({
                        "growth_expectations": {"score": 0.5,  "rationale": "Text mentions moderate expansion."},
                        "inflation_concern":   {"score": 1.0,  "rationale": "Above-target inflation discussed."},
                        "labor_market":        {"score": 1.5,  "rationale": "Near-record low unemployment cited."},
                        "credit_conditions":   {"score": 0.0,  "rationale": "Neutral financial conditions."},
                        "policy_uncertainty":  {"score": -0.5, "rationale": "Path seen as gradual and predictable."},
                    })
                    class Resp:
                        content = [type("C", (), {"text": mock_json})()]
                    return Resp()

        client = MockClient()
        sample = "Economic activity expanded at a moderate pace. Inflation above 2 percent target."
        result = score_document(sample, client, n_runs=2)
        print("Single doc result:")
        for dim, v in result.items():
            if isinstance(v, dict):
                print(f"  {dim:30s}: {v['mean']:+.1f}  (SD={v['std']:.2f})")
        print(f"  reliability_ok: {result['reliability_ok']}")

    else:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        sample = ("Staff Review: Economic activity continued to expand at a moderate pace. "
                  "The labor market remained tight. Inflation was running above 2 percent. "
                  "Financial conditions were broadly accommodative.")
        result = score_document(sample, client, n_runs=LLM_N_RUNS,
                                cache_key="test_doc")
        print("Live API result:")
        for dim in SCORE_DIMS:
            v = result[dim]
            print(f"  {dim:30s}: mean={v['mean']:+.1f}  std={v['std']:.2f}  runs={v['runs']}")
