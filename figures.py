"""
figures.py — Reading the Fed (4-model version, 6 target variables)
Generates three publication-quality figures.

Outputs (saved to ./outputs/):
  figure1_llm_scores.png        — LLM score time series (5 FOMC dims)
  figure2_rmse_comparison.png   — RMSE ratios: AR, BVAR, BVAR+LLM, AR+LLM
  figure3_subperiod_heatmap.png — Sub-period BVAR+LLM gain over BVAR

Usage:  python figures.py
"""

import os, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

main_df   = pd.read_csv(OUT / "results_main.csv")
sub_df    = pd.read_csv(OUT / "results_subperiod.csv")
scores_df = pd.read_csv(OUT / "llm_scores.csv", index_col=0)

# Drop VIX and BAA10Y
KEEP = ['GDPC1_gr','CPIAUCSL_yoy','PCEPILFE_yoy','UNRATE','FEDFUNDS','T10Y2Y']
main_df = main_df[main_df['variable'].isin(KEEP)]
sub_df  = sub_df[sub_df['variable'].isin(KEEP)]

VAR_LABELS = {
    "GDPC1_gr":     "Real GDP growth",
    "CPIAUCSL_yoy": "CPI inflation",
    "PCEPILFE_yoy": "Core PCE",
    "UNRATE":       "Unemployment",
    "FEDFUNDS":     "Fed funds rate",
    "T10Y2Y":       "10Y–2Y spread",
}

BLUE   = "#1a4e8c"
TEAL   = "#0f6e56"
AMBER  = "#d68910"
RED    = "#c0392b"
PURPLE = "#533fad"
GOLD   = "#b7950b"
GREY   = "#7f8c8d"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

def q2num(q):
    yr, qt = str(q).split("Q")
    return int(yr) + (int(qt)-1)/4

scores_df["_x"] = [q2num(q) for q in scores_df.index]

# ── Figure 1: LLM Scores ──────────────────────────────────────────────────────
def fig1_llm_scores():
    dims = [
        ("fomc_minutes_growth_expectations", "Growth expectations",  BLUE),
        ("fomc_minutes_inflation_concern",   "Inflation concern",    RED),
        ("fomc_minutes_labor_market",        "Labor market",         TEAL),
        ("fomc_minutes_credit_conditions",   "Credit conditions",    AMBER),
        ("fomc_minutes_policy_uncertainty",  "Policy uncertainty",   PURPLE),
    ]
    x = scores_df["_x"].values
    regimes = [
        (2008.75, 2009.5,  "#fee2e2", "GFC"),
        (2020.0,  2020.5,  "#fef9c3", "COVID"),
        (2021.5,  2023.5,  "#fde8d8", "Inflation\nsurge"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    fig.subplots_adjust(hspace=0.12, top=0.93, bottom=0.07, left=0.10, right=0.97)
    for ax, (col, label, color) in zip(axes, dims):
        y = scores_df[col].values if col in scores_df.columns else np.full(len(x), np.nan)
        for rs, re, rc, rl in regimes:
            ax.axvspan(rs, re, color=rc, alpha=0.6, zorder=0)
        ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--", alpha=0.5)
        yf = np.array(y, dtype=float)
        ax.fill_between(x, yf, 0, where=yf>0, alpha=0.18, color=color, interpolate=True)
        ax.fill_between(x, yf, 0, where=yf<0, alpha=0.18, color=RED,   interpolate=True)
        ax.plot(x, yf, color=color, linewidth=1.8, zorder=3)
        ax.set_ylim(-2.3, 2.3)
        ax.set_yticks([-2,-1,0,1,2])
        ax.set_ylabel(label, fontsize=9, color=color, fontweight="bold")
        ax.yaxis.set_label_coords(-0.07, 0.5)
    axes[-1].set_xlim(min(x)-0.1, max(x)+0.1)
    year_ticks = list(range(2008, 2025, 2))
    axes[-1].set_xticks(year_ticks)
    axes[-1].set_xticklabels([str(y) for y in year_ticks])
    for rs, re, rc, rl in regimes:
        axes[0].text((rs+re)/2, 2.05, rl, ha="center", va="bottom",
                     fontsize=7.5, color="#555", style="italic")
    fig.suptitle("Figure 1.  LLM Sentiment Scores from FOMC Minutes, 2007Q4–2024Q4",
                 fontsize=11, fontweight="bold", y=0.97)
    fig.text(0.10, 0.01,
             "Note: Scores extracted from FOMC minutes by Claude (Anthropic API) on a −2 to +2 scale. "
             "Shading indicates the Global Financial Crisis (red), COVID shock (yellow), "
             "and inflation surge (orange).", fontsize=7.5, color=GREY)
    fig.savefig(OUT/"figure1_llm_scores.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ Figure 1 saved")


# ── Figure 2: RMSE Comparison (4 models) ─────────────────────────────────────
def fig2_rmse_bars():
    vars_ord = KEEP
    labels   = [VAR_LABELS[v] for v in vars_ord]
    n = len(vars_ord)
    x = np.arange(n)
    w = 0.20   # narrower bars for 4 models

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.38, top=0.88, bottom=0.25, left=0.07, right=0.97)

    for ax, h in zip(axes, [1, 2]):
        sub = main_df[main_df["horizon"] == h].set_index("variable")

        bvar_r  = [sub.loc[v,"BVAR_ratio"]     if v in sub.index else np.nan for v in vars_ord]
        llm_r   = [sub.loc[v,"BVAR_LLM_ratio"] if v in sub.index else np.nan for v in vars_ord]
        arlm_r  = [sub.loc[v,"AR_LLM_ratio"]   if v in sub.index else np.nan for v in vars_ord]
        sigs_bl = [str(sub.loc[v,"BVAR_LLM_dm_sig"]) if v in sub.index else "" for v in vars_ord]
        sigs_al = [str(sub.loc[v,"AR_LLM_dm_sig"])   if v in sub.index else "" for v in vars_ord]

        offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

        # AR bar (= 1.0, reference)
        ax.bar(x + offsets[0], np.ones(n), width=w, color=GREY, alpha=0.55,
               label="AR (= 1.0)", zorder=3)
        # BVAR bar
        ax.bar(x + offsets[1], bvar_r, width=w, color=BLUE, alpha=0.75,
               label="BVAR / AR", zorder=3)
        # BVAR+LLM bar
        bvllm_colors = ["#0f6e56" if (l is not None and b is not None
                         and not np.isnan(l) and l < b) else "#a8d5c4"
                        for l, b in zip(llm_r, bvar_r)]
        ax.bar(x + offsets[2], llm_r, width=w, color=bvllm_colors, alpha=0.85,
               label="BVAR+LLM / AR", zorder=3)
        # AR+LLM bar
        arlm_colors = ["#7d3c98" if (a is not None and not np.isnan(a) and a < 1.0)
                       else "#d7bde2" for a in arlm_r]
        arlm_bars = ax.bar(x + offsets[3], arlm_r, width=w, color=arlm_colors, alpha=0.85,
               label="AR+LLM / AR", zorder=3)

        # AR baseline
        ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", zorder=4)

        # DM significance stars
        for i, (sig, r) in enumerate(zip(sigs_al, arlm_r)):
            if sig not in ("", "nan", "None") and not (isinstance(r, float) and np.isnan(r)):
                ax.text(x[i]+offsets[3], r+0.05, sig,
                        ha="center", va="bottom", fontsize=8, color="#7d3c98", fontweight="bold")
        for i, (sig, r) in enumerate(zip(sigs_bl, llm_r)):
            if sig not in ("", "nan", "None") and not (isinstance(r, float) and np.isnan(r)):
                ax.text(x[i]+offsets[2], r+0.05, sig,
                        ha="center", va="bottom", fontsize=8, color="#c0392b", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("RMSE ratio vs. AR(4) baseline")
        ax.set_title(f"h = {h} quarter{'s' if h>1 else ''} ahead", fontweight="bold", pad=8)
        ax.set_ylim(0, min(ax.get_ylim()[1], 12))
        ax.legend(framealpha=0.85, fontsize=8.5, loc="upper left")

        # shade where AR+LLM < 1
        for i, ar in enumerate(arlm_r):
            if ar is not None and not np.isnan(ar) and ar < 1.0:
                ax.axvspan(x[i]-0.45, x[i]+0.45, color="#ede0f7", alpha=0.35, zorder=0)

    fig.suptitle(
        "Figure 2.  RMSE Ratios Relative to AR(4) Baseline — Out-of-Sample Evaluation 2020Q1–2024Q4",
        fontsize=11, fontweight="bold")
    fig.text(0.07, 0.01,
             "Note: Purple shading = AR+LLM beats AR. Stars denote Diebold-Mariano significance "
             "(purple = AR+LLM vs AR; red = BVAR+LLM vs BVAR): * 10%, ** 5%, *** 1%.",
             fontsize=7.5, color=GREY)
    fig.savefig(OUT/"figure2_rmse_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ Figure 2 saved")


# ── Figure 3: Sub-Period Heatmap ──────────────────────────────────────────────
def fig3_subperiod_heatmap():
    periods  = ["COVID shock", "Inflation surge", "Soft landing"]
    horizons = [1, 2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.35, top=0.85, bottom=0.15, left=0.12, right=0.97)

    for ax, h in zip(axes, horizons):
        matrix   = np.full((len(KEEP), len(periods)), np.nan)
        ann_text = [["" for _ in periods] for _ in KEEP]
        for j, period in enumerate(periods):
            psub = sub_df[(sub_df["period"]==period) & (sub_df["horizon"]==h)]
            bvar = psub[psub["model"]=="BVAR"].set_index("variable")["rmse"]
            llm  = psub[psub["model"]=="BVAR_LLM"].set_index("variable")["rmse"]
            for i, v in enumerate(KEEP):
                if v in bvar.index and v in llm.index:
                    bv, lm = float(bvar[v]), float(llm[v])
                    if bv > 0:
                        pct = (lm-bv)/bv*100
                        matrix[i, j] = pct
                        ann_text[i][j] = f"{pct:+.0f}%"

        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=-60, vmax=20, aspect="auto")
        for i in range(len(KEEP)):
            for j in range(len(periods)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 35 else "black"
                    ax.text(j, i, ann_text[i][j], ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, fontsize=9)
        ax.set_yticks(range(len(KEEP)))
        ax.set_yticklabels([VAR_LABELS[v] for v in KEEP], fontsize=9)
        ax.set_title(f"h = {h} quarter{'s' if h>1 else ''} ahead",
                     fontweight="bold", pad=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("% change in RMSE\n(BVAR+LLM vs BVAR)", fontsize=8)
        cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        "Figure 3.  Sub-Period RMSE Change: BVAR+LLM vs. BVAR\n"
        "(negative = LLM augmentation improves forecast accuracy)",
        fontsize=11, fontweight="bold")
    fig.text(0.12, 0.01,
             "Note: Cell values are (BVAR+LLM RMSE − BVAR RMSE) / BVAR RMSE × 100. "
             "Green = LLM augmentation improves accuracy; red = worsens.",
             fontsize=7.5, color=GREY)
    fig.savefig(OUT/"figure3_subperiod_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ Figure 3 saved")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_llm_scores()
    fig2_rmse_bars()
    fig3_subperiod_heatmap()
    print(f"\nAll figures saved to: {OUT.resolve()}/")
