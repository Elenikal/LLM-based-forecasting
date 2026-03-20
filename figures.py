"""
figures.py
==========
Generates three publication-quality figures for:
  "Reading the Fed: LLM-Augmented Forecasting of U.S. GDP and Financial Indicators"

Outputs (saved to ./outputs/):
  figure1_llm_scores.png        — LLM score time series (5 FOMC dims)
  figure2_rmse_comparison.png   — RMSE ratio bar chart (all variables, h=1 and h=2)
  figure3_subperiod_heatmap.png — Sub-period BVAR+LLM gain heatmap

Usage:
  python figures.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Output dir ────────────────────────────────────────────────────────────────
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "outputs"

def load(name):
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}\nRun the pipeline first: python run.py --live")
    return p

main_df   = pd.read_csv(load("results_main.csv"))
sub_df    = pd.read_csv(load("results_subperiod.csv"))
scores_df = pd.read_csv(load("llm_scores.csv"), index_col=0)

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#1a4e8c"
RED    = "#c0392b"
AMBER  = "#d68910"
TEAL   = "#0f6e56"
PURPLE = "#533fad"
GREY   = "#7f8c8d"
LIGHT  = "#ecf0f1"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})

VAR_LABELS = {
    "GDPC1_gr":     "Real GDP growth",
    "CPIAUCSL_yoy": "CPI inflation",
    "PCEPILFE_yoy": "Core PCE",
    "UNRATE":       "Unemployment",
    "FEDFUNDS":     "Fed funds rate",
    "T10Y2Y":       "10Y–2Y spread",
    "BAA10Y":       "BAA spread",
    "VIXCLS_log":   "log(VIX)",
}

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — LLM Score Time Series
# ─────────────────────────────────────────────────────────────────────────────

def fig1_llm_scores():
    dims = [
        ("fomc_minutes_growth_expectations", "Growth expectations",  BLUE),
        ("fomc_minutes_inflation_concern",   "Inflation concern",    RED),
        ("fomc_minutes_labor_market",        "Labor market",         TEAL),
        ("fomc_minutes_credit_conditions",   "Credit conditions",    AMBER),
        ("fomc_minutes_policy_uncertainty",  "Policy uncertainty",   PURPLE),
    ]

    # Parse quarters to numeric for plotting
    def q2num(q):
        yr, qt = q.split("Q")
        return int(yr) + (int(qt) - 1) / 4

    idx = scores_df.index.tolist()
    x   = [q2num(q) for q in idx]

    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    fig.subplots_adjust(hspace=0.12, top=0.93, bottom=0.07, left=0.10, right=0.97)

    # Recession / regime shading
    regimes = [
        (2008.75, 2009.5,  "#fee2e2", "GFC"),
        (2020.0,  2020.5,  "#fef9c3", "COVID"),
        (2021.5,  2023.5,  "#fde8d8", "Inflation\nsurge"),
    ]

    for ax, (col, label, color) in zip(axes, dims):
        y = scores_df[col].values if col in scores_df.columns else np.full(len(x), np.nan)

        # shade regimes
        for rs, re, rc, rl in regimes:
            ax.axvspan(rs, re, color=rc, alpha=0.6, zorder=0)

        # zero line
        ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--", alpha=0.5)

        # fill under curve
        ax.fill_between(x, y, 0, where=np.array(y) > 0,
                         alpha=0.18, color=color, interpolate=True)
        ax.fill_between(x, y, 0, where=np.array(y) < 0,
                         alpha=0.18, color=RED, interpolate=True)

        ax.plot(x, y, color=color, linewidth=1.8, zorder=3)

        ax.set_ylim(-2.3, 2.3)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_ylabel(label, fontsize=9, color=color, fontweight="bold")
        ax.yaxis.set_label_coords(-0.07, 0.5)

    # x-axis: year ticks
    axes[-1].set_xlim(min(x) - 0.1, max(x) + 0.1)
    year_ticks = list(range(2008, 2025, 2))
    axes[-1].set_xticks(year_ticks)
    axes[-1].set_xticklabels([str(y) for y in year_ticks])

    # Regime labels on top panel
    for rs, re, rc, rl in regimes:
        mid = (rs + re) / 2
        axes[0].text(mid, 2.05, rl, ha="center", va="bottom",
                     fontsize=7.5, color="#555", style="italic")

    fig.suptitle("Figure 1.  LLM Sentiment Scores from FOMC Minutes, 2007Q4–2024Q4",
                 fontsize=11, fontweight="bold", y=0.97)

    note = ("Note: Scores extracted from FOMC minutes by Claude (Anthropic API) on a −2 to +2 scale. "
            "Shading indicates the Global Financial Crisis (red), COVID shock (yellow), "
            "and inflation surge (orange).")
    fig.text(0.10, 0.01, note, fontsize=7.5, color=GREY, wrap=True)

    fig.savefig(OUT / "figure1_llm_scores.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  ✓ Figure 1 saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — RMSE Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def fig2_rmse_bars():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.38, top=0.90, bottom=0.22, left=0.08, right=0.97)

    vars_ordered = ["GDPC1_gr", "CPIAUCSL_yoy", "PCEPILFE_yoy", "UNRATE",
                    "FEDFUNDS", "T10Y2Y", "BAA10Y", "VIXCLS_log"]
    labels = [VAR_LABELS[v] for v in vars_ordered]
    n = len(vars_ordered)
    x = np.arange(n)
    w = 0.28

    for ax, h in zip(axes, [1, 2]):
        sub = main_df[main_df["horizon"] == h].set_index("variable")

        bvar_r = [sub.loc[v, "BVAR_ratio"]    if v in sub.index else np.nan for v in vars_ordered]
        llm_r  = [sub.loc[v, "BVAR_LLM_ratio"] if v in sub.index else np.nan for v in vars_ordered]
        sigs   = [sub.loc[v, "BVAR_LLM_dm_sig"] if v in sub.index else "" for v in vars_ordered]

        bars_bvar = ax.bar(x - w/2, bvar_r, width=w, color=BLUE,
                           alpha=0.75, label="BVAR / AR", zorder=3)
        bars_llm  = ax.bar(x + w/2, llm_r,  width=w, color=TEAL,
                           alpha=0.85, label="BVAR+LLM / AR", zorder=3)

        ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
                   label="AR baseline (= 1.0)", zorder=4)

        # DM significance stars above LLM bar
        for i, (r, sig) in enumerate(zip(llm_r, sigs)):
            if pd.notna(sig) and sig not in ("", "nan"):
                ax.text(x[i] + w/2, r + 0.15, str(sig),
                        ha="center", va="bottom", fontsize=8.5,
                        color="#c0392b", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
        ax.set_ylabel("RMSE ratio vs. AR(4) baseline")
        ax.set_title(f"h = {h} quarter{'s' if h > 1 else ''} ahead",
                     fontweight="bold", pad=8)
        ax.legend(framealpha=0.85, fontsize=8.5)

        # Clip y-axis to show detail (T10Y2Y and VIX have very large ratios)
        ax.set_ylim(0, min(ax.get_ylim()[1], 22))

        # Shade variables where LLM < BVAR
        for i, (br, lr) in enumerate(zip(bvar_r, llm_r)):
            if pd.notna(lr) and pd.notna(br) and lr < br:
                ax.axvspan(x[i] - 0.45, x[i] + 0.45, color="#d5f5e3",
                           alpha=0.35, zorder=0)

    fig.suptitle(
        "Figure 2.  RMSE Ratios Relative to AR(4) Baseline — "
        "Out-of-Sample Evaluation 2020Q1–2024Q4",
        fontsize=11, fontweight="bold")
    note = ("Note: Green shading = BVAR+LLM beats BVAR. Stars above BVAR+LLM bars denote "
            "Diebold-Mariano significance: * 10%, ** 5%, *** 1%.")
    fig.text(0.08, 0.01, note, fontsize=7.5, color=GREY)

    fig.savefig(OUT / "figure2_rmse_comparison.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  ✓ Figure 2 saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Sub-Period Heatmap of BVAR+LLM Gain over BVAR
# ─────────────────────────────────────────────────────────────────────────────

def fig3_subperiod_heatmap():
    periods   = ["COVID shock", "Inflation surge", "Soft landing"]
    vars_ord  = ["GDPC1_gr", "CPIAUCSL_yoy", "PCEPILFE_yoy", "UNRATE",
                 "FEDFUNDS", "T10Y2Y", "BAA10Y", "VIXCLS_log"]
    horizons  = [1, 2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.15, left=0.12, right=0.97)

    for ax, h in zip(axes, horizons):
        # Build matrix: rows = variables, cols = periods
        # Cell value = (BVAR_LLM - BVAR) / BVAR  → negative = LLM wins
        matrix = np.full((len(vars_ord), len(periods)), np.nan)
        for j, period in enumerate(periods):
            psub = sub_df[(sub_df["period"] == period) & (sub_df["horizon"] == h)]
            bvar_rmse = psub[psub["model"] == "BVAR"].set_index("variable")["rmse"]
            llm_rmse  = psub[psub["model"] == "BVAR_LLM"].set_index("variable")["rmse"]
            for i, v in enumerate(vars_ord):
                if v in bvar_rmse.index and v in llm_rmse.index:
                    bv = bvar_rmse[v]
                    lm = llm_rmse[v]
                    if bv > 0:
                        matrix[i, j] = (lm - bv) / bv * 100  # % change

        # Colour: negative (LLM wins) = green, positive (LLM loses) = red
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=-60, vmax=20,
                       aspect="auto")

        # Cell annotations
        for i in range(len(vars_ord)):
            for j in range(len(periods)):
                val = matrix[i, j]
                if not np.isnan(val):
                    txt = f"{val:+.0f}%"
                    color = "white" if abs(val) > 35 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=8.5, fontweight="bold", color=color)

        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, fontsize=9)
        ax.set_yticks(range(len(vars_ord)))
        ax.set_yticklabels([VAR_LABELS[v] for v in vars_ord], fontsize=9)
        ax.set_title(f"h = {h} quarter{'s' if h > 1 else ''} ahead",
                     fontweight="bold", pad=8)

        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("% change in RMSE\n(BVAR+LLM vs BVAR)", fontsize=8)
        cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        "Figure 3.  Sub-Period RMSE Change: BVAR+LLM vs. BVAR\n"
        "(negative = LLM augmentation improves forecast accuracy)",
        fontsize=11, fontweight="bold")
    note = ("Note: Cell values are (BVAR+LLM RMSE − BVAR RMSE) / BVAR RMSE × 100. "
            "Green = LLM augmentation improves accuracy; red = worsens.")
    fig.text(0.12, 0.01, note, fontsize=7.5, color=GREY)

    fig.savefig(OUT / "figure3_subperiod_heatmap.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  ✓ Figure 3 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    fig1_llm_scores()
    fig2_rmse_bars()
    fig3_subperiod_heatmap()
    print(f"\nAll figures saved to: {OUT.resolve()}/")
