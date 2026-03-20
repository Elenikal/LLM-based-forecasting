"""
evaluation/metrics.py
=====================
Forecast evaluation: RMSE, MAE, and the Diebold-Mariano (1995) test
with Harvey-Leybourne-Newbold (1997) small-sample correction.

Also provides:
- RMSE ratio tables (relative to AR baseline)
- Sub-period breakdown (COVID, inflation surge, soft landing)
- Formatting helpers for LaTeX output
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SIGNIFICANCE


# ─────────────────────────────────────────────────────────────────────────────
# Point forecast accuracy
# ─────────────────────────────────────────────────────────────────────────────

def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root mean squared error."""
    err = np.asarray(actual) - np.asarray(forecast)
    return float(np.sqrt(np.mean(err ** 2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error."""
    err = np.asarray(actual) - np.asarray(forecast)
    return float(np.mean(np.abs(err)))


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean forecast error (positive = over-prediction)."""
    return float(np.mean(np.asarray(forecast) - np.asarray(actual)))


# ─────────────────────────────────────────────────────────────────────────────
# Diebold-Mariano test with HLN small-sample correction
# ─────────────────────────────────────────────────────────────────────────────

def diebold_mariano(actual: np.ndarray, fc1: np.ndarray, fc2: np.ndarray,
                    h: int = 1, loss: str = "squared",
                    alternative: str = "two-sided") -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy,
    with Harvey-Leybourne-Newbold (1997) small-sample correction.

    H0: equal predictive accuracy (E[d_t] = 0)
    H1 (two-sided): model 1 and model 2 differ
    H1 (less)     : model 1 is worse than model 2
    H1 (greater)  : model 1 is better than model 2

    Parameters
    ----------
    actual  : (T,) realised values
    fc1     : (T,) forecasts from model 1
    fc2     : (T,) forecasts from model 2
    h       : forecast horizon (for HAC bandwidth and HLN correction)
    loss    : "squared" or "absolute"
    alternative : "two-sided", "less", "greater"

    Returns
    -------
    dict with keys: stat (float), pvalue (float), dm_stat (float, uncorrected),
                    mean_d (float), n (int)
    """
    a  = np.asarray(actual)
    f1 = np.asarray(fc1)
    f2 = np.asarray(fc2)

    e1 = a - f1
    e2 = a - f2

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    T = len(d)
    mean_d = np.mean(d)

    # HAC variance estimate (Newey-West with bandwidth h)
    gamma0 = np.var(d, ddof=0)
    acov   = sum(
        (1 - lag / (h + 1)) * np.cov(d[lag:], d[:-lag])[0, 1]
        for lag in range(1, h + 1)
        if len(d[lag:]) > 1
    ) if h > 0 else 0.0

    var_d = (gamma0 + 2 * acov) / T
    if var_d <= 0:
        var_d = gamma0 / T + 1e-12

    # DM statistic
    dm_stat = mean_d / np.sqrt(var_d)

    # HLN correction: multiply by sqrt((T + 1 - 2h + h(h-1)/T) / T)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_corrected = dm_stat * correction

    # t-distribution with T-1 degrees of freedom
    if alternative == "two-sided":
        pvalue = 2 * stats.t.sf(np.abs(dm_corrected), df=T - 1)
    elif alternative == "less":
        pvalue = stats.t.cdf(dm_corrected, df=T - 1)
    elif alternative == "greater":
        pvalue = stats.t.sf(dm_corrected, df=T - 1)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return {
        "stat":        float(dm_corrected),
        "dm_stat":     float(dm_stat),
        "pvalue":      float(pvalue),
        "mean_d":      float(mean_d),
        "n":           T,
        "reject_h0":   pvalue < SIGNIFICANCE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results table construction
# ─────────────────────────────────────────────────────────────────────────────

def build_results_table(forecasts: dict, actuals: dict,
                        target_vars: list, horizons: list,
                        baseline_model: str = "AR") -> pd.DataFrame:
    """
    Build a summary results table.

    Parameters
    ----------
    forecasts : {model_name: {target: {horizon: np.ndarray}}}
    actuals   : {target: np.ndarray}
    target_vars : list of variable names
    horizons    : list of forecast horizons
    baseline_model : name of the model to use as RMSE denominator

    Returns
    -------
    DataFrame with MultiIndex columns (model, metric) and rows (target, horizon)
    """
    rows = []
    models = list(forecasts.keys())

    for var in target_vars:
        act = actuals.get(var)
        if act is None:
            continue
        for h in horizons:
            row = {"variable": var, "horizon": h}
            baseline_rmse_val = None

            for model in models:
                fc = forecasts[model].get(var, {}).get(h)
                if fc is None or act is None:
                    row[f"{model}_rmse"] = np.nan
                    row[f"{model}_mae"]  = np.nan
                    continue

                r = rmse(act, fc)
                m = mae(act, fc)
                row[f"{model}_rmse"] = r
                row[f"{model}_mae"]  = m

                if model == baseline_model:
                    baseline_rmse_val = r

            # Compute RMSE ratios relative to baseline
            if baseline_rmse_val and baseline_rmse_val > 0:
                for model in models:
                    r = row.get(f"{model}_rmse")
                    if r is not None and not np.isnan(r):
                        row[f"{model}_ratio"] = r / baseline_rmse_val

            # DM tests vs baseline for each non-baseline model
            act_arr  = np.asarray(act)
            base_fc  = forecasts.get(baseline_model, {}).get(var, {}).get(h)
            if base_fc is not None:
                for model in models:
                    if model == baseline_model:
                        continue
                    fc = forecasts[model].get(var, {}).get(h)
                    if fc is not None and len(fc) == len(base_fc):
                        dm = diebold_mariano(act_arr, np.asarray(base_fc),
                                             np.asarray(fc), h=h)
                        row[f"{model}_dm_pval"] = dm["pvalue"]
                        row[f"{model}_dm_sig"]  = (
                            "***" if dm["pvalue"] < 0.01 else
                            "**"  if dm["pvalue"] < 0.05 else
                            "*"   if dm["pvalue"] < 0.10 else ""
                        )

            rows.append(row)

    return pd.DataFrame(rows).set_index(["variable", "horizon"])


def sub_period_results(forecasts: dict, actuals: dict,
                       quarters: list, target_vars: list,
                       horizons: list,
                       sub_periods: Optional[dict] = None) -> dict:
    """
    Compute RMSE for each model within defined sub-periods.

    Parameters
    ----------
    quarters    : list of quarter strings aligned with forecast arrays
    sub_periods : {name: (start_q, end_q)} e.g.
                  {"COVID": ("2020Q1","2021Q2"), "Inflation": ("2021Q3","2023Q2")}
    """
    if sub_periods is None:
        sub_periods = {
            "COVID shock":       ("2020Q1", "2021Q2"),
            "Inflation surge":   ("2021Q3", "2023Q2"),
            "Soft landing":      ("2023Q3", "2024Q4"),
        }

    q_idx = {q: i for i, q in enumerate(quarters)}
    results = {}

    for period_name, (start_q, end_q) in sub_periods.items():
        idx_start = q_idx.get(start_q, 0)
        idx_end   = q_idx.get(end_q, len(quarters) - 1) + 1
        mask      = slice(idx_start, idx_end)

        period_rmse = {}
        for model, model_fc in forecasts.items():
            period_rmse[model] = {}
            for var in target_vars:
                period_rmse[model][var] = {}
                act = actuals.get(var)
                if act is None:
                    continue
                for h in horizons:
                    fc = model_fc.get(var, {}).get(h)
                    if fc is None:
                        continue
                    try:
                        r = rmse(np.asarray(act)[mask], np.asarray(fc)[mask])
                        period_rmse[model][var][h] = r
                    except Exception:
                        pass

        results[period_name] = period_rmse

    return results


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table formatter
# ─────────────────────────────────────────────────────────────────────────────

def to_latex_table(df: pd.DataFrame, caption: str = "",
                   label: str = "tab:results") -> str:
    """
    Convert a results DataFrame to a LaTeX table string.
    RMSE ratios < 1 printed in bold (improvement over baseline).
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{ll" + "r" * (len(df.columns)) + "}",
        r"\toprule",
    ]

    # Header
    header = "Variable & Horizon & " + " & ".join(df.columns.tolist()) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for (var, h), row in df.iterrows():
        cells = [str(var), f"h={h}"]
        for col, val in row.items():
            if pd.isna(val):
                cells.append("—")
            elif isinstance(val, float):
                if "ratio" in col and val < 1.0:
                    cells.append(f"\\textbf{{{val:.3f}}}")
                else:
                    cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    T = 20
    actual = np.random.randn(T)
    fc1    = actual + np.random.randn(T) * 0.8   # decent model
    fc2    = actual + np.random.randn(T) * 1.5   # worse model

    print("RMSE model1:", rmse(actual, fc1))
    print("RMSE model2:", rmse(actual, fc2))
    print("MAE  model1:", mae(actual, fc1))

    dm = diebold_mariano(actual, fc2, fc1, h=1)
    print(f"\nDM test (fc2 vs fc1, H0: equal accuracy):")
    print(f"  DM stat (corrected) = {dm['stat']:.3f}")
    print(f"  p-value             = {dm['pvalue']:.3f}")
    print(f"  mean loss diff      = {dm['mean_d']:.4f}")
    print(f"  reject H0 at 10%    = {dm['reject_h0']}")

    # Synthetic table
    forecasts = {
        "AR":   {"GDPC1_gr": {1: fc2, 2: fc2}},
        "BVAR": {"GDPC1_gr": {1: fc1, 2: fc1}},
    }
    actuals = {"GDPC1_gr": actual}
    tbl = build_results_table(forecasts, actuals, ["GDPC1_gr"], [1, 2])
    print("\nResults table:")
    print(tbl.to_string())
