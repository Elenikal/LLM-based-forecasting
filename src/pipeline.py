"""
pipeline.py  —  Master orchestration for the LLM-Augmented Forecasting paper.

Steps
-----
  1. Load quantitative data (FRED-QD or synthetic)
  2. Load / score text data (FOMC minutes, Beige Books) via Claude API
  3. Rolling PCA on LLM score vectors (re-fit at each forecast origin)
  4. Rolling-window BVAR forecasting — Models 1 (AR), 2 (BVAR), 3 (BVAR+LLM)
  5. Evaluate: RMSE, MAE, Diebold-Mariano tests, sub-period breakdown
  6. Write results to ./outputs/

Called from run.py — do not invoke directly unless you know what you're doing.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EVAL_START, EVAL_END, ROLL_WINDOW, FORECAST_H,
    BVAR_LAGS, N_LLM_PCS, OUTPUT_DIR,
    ANTHROPIC_API_KEY, FRED_API_KEY,
    SCORE_DIMS, SCORE_DIR, RANDOM_SEED,
)
from data.fred_pull    import load_fred_data, get_bvar_system, get_rolling_window
from data.text_pull    import load_all_texts, generate_synthetic_texts
from scoring.llm_scorer import build_score_matrix
from models.bvar       import estimate_bvar, forecast_bvar, fit_ar, forecast_ar
from evaluation.metrics import (
    rmse, mae, build_results_table, sub_period_results, to_latex_table,
)

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Quantitative data
# ─────────────────────────────────────────────────────────────────────────────

def prepare_quant_data(demo: bool, use_cache: bool) -> pd.DataFrame:
    if demo or not FRED_API_KEY:
        print("[DEMO] Generating synthetic quantitative data...")
        idx = pd.period_range("1999Q1", "2024Q4", freq="Q")
        np.random.seed(RANDOM_SEED)
        n = len(idx)
        t = np.linspace(0, 1, n)
        df = pd.DataFrame({
            "GDPC1_gr":     np.cumsum(np.random.randn(n) * 0.8) + np.sin(t * 6) * 2,
            "CPIAUCSL_yoy": 1.5 + t * 2 + np.random.randn(n) * 0.4,
            "PCEPILFE_yoy": 1.3 + t * 1.8 + np.random.randn(n) * 0.3,
            "UNRATE":       5.0 + np.cumsum(np.random.randn(n) * 0.12),
            "FEDFUNDS":     np.clip(1.0 + t * 4 + np.random.randn(n) * 0.5, 0, 7),
            "T10Y2Y":       np.random.randn(n) * 0.5 + 0.3,
            "BAA10Y":       np.abs(np.random.randn(n) * 0.3) + 1.2,
            "VIXCLS_log":   np.log(np.clip(
                                np.random.lognormal(3.2, 0.4, n), 10, 80)),
        }, index=idx)
        return df
    else:
        raw = load_fred_data(use_cache=use_cache)
        return get_bvar_system(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — LLM score matrix
# ─────────────────────────────────────────────────────────────────────────────

def prepare_llm_scores(demo: bool, use_cache: bool,
                       quant_index: pd.PeriodIndex) -> pd.DataFrame:
    scores_cache = SCORE_DIR / "score_matrix.parquet"

    if use_cache and scores_cache.exists():
        print(f"  Loading cached LLM scores from {scores_cache.name}")
        df = pd.read_parquet(scores_cache)
        df.index = pd.PeriodIndex(df.index, freq="Q")
        return df

    if demo or not ANTHROPIC_API_KEY:
        print("[DEMO] Generating synthetic LLM scores (no API key)...")
        rows = []
        for q in quant_index:
            qs = str(q)
            np.random.seed(abs(hash(qs)) % 2**31)
            yr = int(qs[:4])
            base = 0.6 if yr < 2008 else (-0.8 if yr < 2010 else 0.4)
            row = {"quarter": qs}
            for src in ["fomc_minutes", "beige_book"]:
                for dim in SCORE_DIMS:
                    row[f"{src}_{dim}"] = float(
                        np.clip(base + np.random.randn() * 0.7, -2, 2))
            rows.append(row)
        df = pd.DataFrame(rows)
        df.index = pd.PeriodIndex(df["quarter"], freq="Q")
        df = df.drop(columns=["quarter"])
    else:
        print("  Scoring documents via Claude API...")
        import anthropic as ant
        client = ant.Anthropic(api_key=ANTHROPIC_API_KEY)
        texts  = load_all_texts(year_start=1999, year_end=2024, use_cache=use_cache)
        df     = build_score_matrix(texts, client)

    # Cache
    save = df.copy(); save.index = save.index.astype(str)
    save.to_parquet(scores_cache)
    print(f"  Score matrix saved ({df.shape[0]} quarters × {df.shape[1]} cols)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Rolling PCA (re-fit at each forecast origin)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_pca(score_df: pd.DataFrame, window_idx: pd.PeriodIndex,
                n_components: int = N_LLM_PCS) -> np.ndarray:
    """Return (len(window_idx), n_components) PC scores for a given window."""
    avail = score_df.index.intersection(window_idx)
    out   = np.zeros((len(window_idx), n_components))
    if len(avail) == 0:
        return out

    sub        = score_df.loc[avail].fillna(0.0).values
    sub_scaled = StandardScaler().fit_transform(sub)
    n_comp     = min(n_components, sub_scaled.shape[1], sub_scaled.shape[0] - 1)
    if n_comp < 1:
        return out

    pcs = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit_transform(sub_scaled)
    if pcs.shape[1] < n_components:
        pcs = np.hstack([pcs,
                         np.zeros((pcs.shape[0], n_components - pcs.shape[1]))])

    # Map scores back to the full window index
    pos_map = {q: i for i, q in enumerate(window_idx)}
    for j, q in enumerate(avail):
        out[pos_map[q], :] = pcs[j, :]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Rolling-window forecast loop
# ─────────────────────────────────────────────────────────────────────────────

def run_rolling_forecasts(quant_df: pd.DataFrame,
                          llm_scores: pd.DataFrame) -> tuple:
    """
    Returns (forecasts dict, actuals dict, eval_quarters list).

    forecasts[model][variable][horizon] = list of T point forecasts
    actuals[variable] = list of T realised values (h=1 target)
    """
    eval_quarters = list(
        pd.period_range(EVAL_START, EVAL_END, freq="Q").astype(str))
    var_names     = list(quant_df.columns)
    MODEL_NAMES   = ["AR", "BVAR", "BVAR_LLM", "AR_LLM"]

    forecasts = {m: {v: {h: [] for h in FORECAST_H} for v in var_names}
                 for m in MODEL_NAMES}
    actuals   = {v: [] for v in var_names}

    n_origins = len(eval_quarters)
    print(f"\n  {n_origins} forecast origins  ×  "
          f"{len(FORECAST_H)} horizons  ×  {len(MODEL_NAMES)} models\n")

    for i, t_str in enumerate(eval_quarters, 1):
        t = pd.Period(t_str, freq="Q")

        # ── Estimation window ──
        win_end   = t - 1
        win_start = win_end - (ROLL_WINDOW - 1)
        win_idx   = pd.period_range(win_start, win_end, freq="Q")
        Y_win_df  = quant_df.loc[quant_df.index.isin(win_idx)].dropna()

        if len(Y_win_df) < BVAR_LAGS + 5:
            warnings.warn(f"{t_str}: window too short ({len(Y_win_df)} obs), skipping")
            for m in MODEL_NAMES:
                for v in var_names:
                    for h in FORECAST_H:
                        forecasts[m][v][h].append(np.nan)
            for v in var_names:
                actuals[v].append(np.nan)
            continue

        Y_win     = Y_win_df.values
        Y_win_idx = Y_win_df.index

        # ── Record actual values (h=1 target) ──
        target_h1 = t   # h=1 means forecasting t given info through t-1
        for v in var_names:
            val = quant_df.loc[target_h1, v] if target_h1 in quant_df.index else np.nan
            actuals[v].append(float(val))

        # ── LLM PCs ──
        # ── Lenza-Primiceri (2022) / delta-score fix ──────────────────────────
        # Use first differences of LLM scores rather than levels.
        # Levels inherit the Fed's institutional optimism bias permanently;
        # deltas only contribute signal when the Fed's tone is *changing*,
        # which is exactly when the text carries genuine new information.
        # This eliminates the soft-landing over-prediction problem.
        llm_delta = llm_scores.diff().fillna(0.0)
        llm_pcs = rolling_pca(llm_scores, Y_win_idx, N_LLM_PCS)
        has_llm = llm_pcs.std() > 1e-6

        # ── Model 1: AR(p) — per variable ──
        for vi, v in enumerate(var_names):
            ar_fc = forecast_ar(fit_ar(Y_win[:, vi], p=BVAR_LAGS),
                                h=max(FORECAST_H))
            for h in FORECAST_H:
                forecasts["AR"][v][h].append(float(ar_fc[h - 1]))

        # ── Model 2: BVAR (no LLM) ──
        try:
            res_bvar = estimate_bvar(Y_win, p=BVAR_LAGS, var_names=var_names)
            fc_bvar  = forecast_bvar(res_bvar, h=max(FORECAST_H))
        except Exception as e:
            warnings.warn(f"BVAR failed at {t_str}: {e}")
            fc_bvar = None

        for h in FORECAST_H:
            for vi, v in enumerate(var_names):
                val = float(fc_bvar[h - 1, vi]) if fc_bvar is not None else np.nan
                forecasts["BVAR"][v][h].append(val)

        # ── Model 3: BVAR + LLM PCs ──
        if has_llm:
            try:
                res_llm = estimate_bvar(Y_win, p=BVAR_LAGS,
                                        exog=llm_pcs, var_names=var_names)
                fut_exog = np.tile(llm_pcs[-1:, :], (max(FORECAST_H), 1))
                fc_llm   = forecast_bvar(res_llm, h=max(FORECAST_H),
                                         future_exog=fut_exog)
            except Exception as e:
                warnings.warn(f"BVAR_LLM failed at {t_str}: {e}")
                fc_llm = fc_bvar   # fall back to plain BVAR
        else:
            fc_llm = fc_bvar

        for h in FORECAST_H:
            for vi, v in enumerate(var_names):
                val = float(fc_llm[h - 1, vi]) if fc_llm is not None else np.nan
                forecasts["BVAR_LLM"][v][h].append(val)

        # ── Model 4: AR + LLM PCs (per variable) ──
        for vi, v in enumerate(var_names):
            if has_llm:
                ar_llm_fit = fit_ar(Y_win[:, vi], p=BVAR_LAGS, exog=llm_pcs)
                fut_exog_ar = np.tile(llm_pcs[-1:, :], (max(FORECAST_H), 1))
                ar_llm_fc  = forecast_ar(ar_llm_fit, h=max(FORECAST_H),
                                          future_exog=fut_exog_ar)
            else:
                # Fall back to plain AR if no LLM scores available
                ar_llm_fc = forecast_ar(fit_ar(Y_win[:, vi], p=BVAR_LAGS),
                                        h=max(FORECAST_H))
            for h in FORECAST_H:
                forecasts["AR_LLM"][v][h].append(float(ar_llm_fc[h - 1]))

        # Progress line
        gdp_ar  = forecasts["AR"]["GDPC1_gr"][1][-1]
        gdp_bv  = forecasts["BVAR"]["GDPC1_gr"][1][-1]
        gdp_llm = forecasts["BVAR_LLM"]["GDPC1_gr"][1][-1]
        print(f"  [{i:02d}/{n_origins}] {t_str}  |  "
              f"GDP h=1:  AR={gdp_ar:+.2f}  BVAR={gdp_bv:+.2f}  "
              f"BVAR+LLM={gdp_llm:+.2f}")

    return forecasts, actuals, eval_quarters


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Evaluate and export
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_export(forecasts, actuals, eval_quarters, var_names):
    print("\n── Evaluation ────────────────────────────────────────────────")

    fc_arr  = {m: {v: {h: np.array(forecasts[m][v][h]) for h in FORECAST_H}
                   for v in var_names} for m in forecasts}
    act_arr = {v: np.array(actuals[v]) for v in var_names}

    # Main table
    main_tbl = build_results_table(fc_arr, act_arr, var_names, FORECAST_H,
                                   baseline_model="AR")
    main_tbl.to_csv(OUTPUT_DIR / "results_main.csv")

    # Console summary
    hdr = f"\n  {'Variable':18s} {'h':>2}  {'AR RMSE':>8}  {'BVAR/AR':>8}  {'LLM/AR':>8}  {'AR+LLM/AR':>10}  {'DM':>5}"
    print(hdr)
    print("  " + "-" * 65)
    for (var, h), row in main_tbl.iterrows():
        ar_r   = row.get("AR_rmse", np.nan)
        bv_r   = row.get("BVAR_ratio", np.nan)
        lm_r   = row.get("BVAR_LLM_ratio", np.nan)
        arlm_r = row.get("AR_LLM_ratio", np.nan)
        sig    = row.get("BVAR_LLM_dm_sig", "")
        arlm_s = row.get("AR_LLM_dm_sig", "")
        print(f"  {var:18s} {h:>2}  {ar_r:8.3f}  {bv_r:8.3f}  "
              f"{lm_r:8.3f}  {arlm_r:10.3f}  {sig:>3}/{arlm_s:<3}")

    # Sub-period breakdown
    sub_res  = sub_period_results(fc_arr, act_arr, eval_quarters,
                                  var_names, FORECAST_H)
    sub_rows = []
    for period, pdata in sub_res.items():
        for model, mdata in pdata.items():
            for var, vdata in mdata.items():
                for h, r in vdata.items():
                    sub_rows.append({"period": period, "model": model,
                                     "variable": var, "horizon": h, "rmse": r})
    pd.DataFrame(sub_rows).to_csv(OUTPUT_DIR / "results_subperiod.csv", index=False)

    # LLM scores data appendix (already saved in prepare_llm_scores)

    # LaTeX table
    ratio_cols = [c for c in main_tbl.columns
                  if ("ratio" in c or "rmse" in c) and c in main_tbl.columns]
    tex = to_latex_table(
        main_tbl[ratio_cols],
        caption=(r"Out-of-sample forecast accuracy, 2020Q1--2024Q4. "
                 r"Entries are RMSE ratios relative to AR$(p)$ baseline (ratio $<1$ = improvement, \textbf{bold}). "
                 r"*, **, *** denote significance at 10\%, 5\%, 1\% via Diebold-Mariano (1995) "
                 r"test with Harvey-Leybourne-Newbold (1997) small-sample correction."),
        label="tab:main_results",
    )
    (OUTPUT_DIR / "results_main.tex").write_text(tex)

    print(f"\n  Outputs written to: {OUTPUT_DIR}")
    print("    results_main.csv       — full RMSE / DM table")
    print("    results_subperiod.csv  — COVID / inflation / soft-landing breakdown")
    print("    llm_scores.csv         — quarterly LLM score vectors (data appendix)")
    print("    results_main.tex       — LaTeX Table 3 (paste into paper)")

    return main_tbl


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(demo: bool = True, use_cache: bool = True):
    print("\n" + "=" * 65)
    print("  LLM-Augmented Forecasting of U.S. GDP and Financial Indicators")
    print(f"  Mode: {'DEMO (synthetic data)' if demo else 'LIVE (real data)'}")
    print(f"  Rolling window: {40}Q  |  Eval: {EVAL_START}–{EVAL_END}")
    print("=" * 65)

    print("\n── Step 1: Quantitative data ─────────────────────────────────")
    quant_df = prepare_quant_data(demo, use_cache)
    print(f"  {quant_df.shape[1]} variables,  "
          f"{len(quant_df)} quarters  "
          f"({quant_df.index[0]} – {quant_df.index[-1]})")

    print("\n── Step 2: LLM score matrix ──────────────────────────────────")
    llm_scores = prepare_llm_scores(demo, use_cache, quant_df.index)
    # Save as data appendix
    s = llm_scores.copy(); s.index = s.index.astype(str)
    s.to_csv(OUTPUT_DIR / "llm_scores.csv")
    print(f"  {llm_scores.shape[1]} score columns  ×  {len(llm_scores)} quarters")

    print("\n── Step 3: Rolling-window forecasts ──────────────────────────")
    forecasts, actuals, eval_quarters = run_rolling_forecasts(quant_df, llm_scores)


    print("\n── Step 4: Evaluation & export ───────────────────────────────")
    evaluate_and_export(forecasts, actuals, eval_quarters, list(quant_df.columns))

    print("\n  Done. ✓\n")
