#!/usr/bin/env python3
"""
live/make_post.py
=================
Post-ready assets for one recorded forecast (Substack does not render
Markdown tables):

  live/posts/YYYY-MM-DD/table.png   table image to paste into the post
  live/posts/YYYY-MM-DD/post.md     plain text: date, proof link, sources

Reads only the recorded files; never writes to forecasts/.

  python live/make_post.py 2026-10-14
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LIVE = Path(__file__).resolve().parent

# Headline set, declared 2026-10-14 before any outcome: the variables where the
# backtest and the COVID robustness check show value in the text. GDP is still
# forecast, recorded and scored, but not headlined.
HEADLINE = ["CPIAUCSL_yoy", "PCEPILFE_yoy", "UNRATE", "FEDFUNDS", "T10Y2Y"]
REPO = "https://github.com/Elenikal/LLM-based-forecasting"

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
ACCENT, RULE, SURFACE = "#2a78d6", "#d9d8d3", "#fcfcfb"

SHORT = {
    "GDPC1_gr":     ("Real GDP growth", "% q/q saar"),
    "CPIAUCSL_yoy": ("CPI inflation", "% y/y"),
    "PCEPILFE_yoy": ("Core PCE inflation", "% y/y"),
    "UNRATE":       ("Unemployment rate", "%"),
    "FEDFUNDS":     ("Fed funds rate", "% effective"),
    "T10Y2Y":       ("10Y-2Y spread", "pp"),
}


def bench_text(bench: pd.DataFrame, var: str, q: str) -> str:
    b = bench[(bench.variable == var) & (bench.target_quarter == q)]
    names = {"Atlanta Fed GDPNow": "GDPNow",
             "Cleveland Fed inflation nowcast": "Cleveland Fed",
             "Fed funds futures (Investing.com Fed Rate Monitor)": "Futures"}
    return "  ·  ".join(f"{names.get(s, s.split(' (')[0].replace(' median', ''))} {v:g}"
                        for s, v in zip(b.source, b.value)) or "–"


def table_png(rows: list[dict], bench: pd.DataFrame, issued: str, out: Path):
    cols = [("", 0.012, "left"), ("Latest", 0.255, "right"), ("Target", 0.345, "right"),
            ("Call", 0.445, "right"), ("68% range", 0.585, "right"),
            ("AR only", 0.675, "right"), ("Outside forecasts", 0.712, "left")]
    n = len(rows)
    fig_h = 2.05 + 0.36 * n
    fig = plt.figure(figsize=(9.6, fig_h), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, fig_h)

    y = fig_h - 0.42
    ax.text(0.012, y, "Reading the Fed", fontsize=15, weight="bold", color=INK, va="center")
    ax.text(0.988, y, f"Forecast issued {issued}", fontsize=10, color=INK_2,
            va="center", ha="right")
    y -= 0.40
    ax.text(0.012, y, "Call = AR(4) + three components of LLM-scored FOMC minutes and Beige Books. "
            "AR only = same model without the text.", fontsize=7.6, color=INK_2, va="center")
    y -= 0.38
    for name, x, ha in cols:
        ax.text(x, y, name, fontsize=8, color=INK_3, ha=ha, va="center")
    y -= 0.16
    ax.plot([0.012, 0.988], [y, y], color=INK_3, lw=0.8)

    prev = None
    for r in rows:
        y -= 0.36
        if r["variable"] != prev and prev is not None:
            ax.plot([0.012, 0.988], [y + 0.18, y + 0.18], color=RULE, lw=0.6)
        name, unit = SHORT[r["variable"]]
        if r["variable"] != prev:
            ax.text(0.012, y, name, fontsize=9, color=INK, va="center")
            ax.text(0.012, y - 0.15, unit, fontsize=6.8, color=INK_3, va="center")
            ax.text(0.255, y, f"{r['last_obs_value']:.2f}", fontsize=9, color=INK_2,
                    ha="right", va="center")
            ax.text(0.255, y - 0.15, r["last_obs_quarter"], fontsize=6.8, color=INK_3,
                    ha="right", va="center")
        ax.text(0.345, y, r["target_quarter"], fontsize=9, color=INK_2, ha="right", va="center")
        ax.text(0.445, y, f"{r['AR_LLM']:.2f}", fontsize=10, weight="bold", color=ACCENT,
                ha="right", va="center")
        ax.text(0.585, y, f"{r['AR_LLM_lo68']:.2f} to {r['AR_LLM_hi68']:.2f}", fontsize=8.5,
                color=INK_2, ha="right", va="center")
        ax.text(0.675, y, f"{r['AR']:.2f}", fontsize=9, color=INK_2, ha="right", va="center")
        ax.text(0.712, y, bench_text(bench, r["variable"], r["target_quarter"]), fontsize=8,
                color=INK_2, va="center")
        prev = r["variable"]

    y -= 0.30
    ax.plot([0.012, 0.988], [y, y], color=INK_3, lw=0.8)
    ax.text(0.012, y - 0.22, f"Timestamped: git tag fc-{issued} · {REPO.removeprefix('https://')}",
            fontsize=7, color=INK_3, va="center")
    ax.text(0.988, y - 0.22, "GDP also recorded and scored in the repo · first-release data",
            fontsize=7, color=INK_3, ha="right", va="center")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    issued = sys.argv[1]
    raw = (LIVE / "forecasts" / f"{issued}.json").read_bytes()
    rec, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
    bf = LIVE / "benchmarks" / f"{issued}.csv"
    bench = pd.read_csv(bf) if bf.exists() else pd.DataFrame(
        columns=["variable", "target_quarter", "source", "value", "as_of", "url"])
    out = LIVE / "posts" / issued
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in rec["forecasts"] if r["variable"] in HEADLINE]
    table_png(rows, bench, issued, out / "table.png")

    docs = rec["meta"]["latest_documents"]
    srcs = bench.drop_duplicates("source")
    lines = [
        f"Forecast issued {issued}.",
        "",
        "[paste table.png here]",
        "",
        f"Latest Fed text in the model: minutes of the {docs['fomc_minutes_meeting']} "
        f"FOMC meeting and the Beige Book of {docs['beige_book_release']}.",
        f"Proof of date: {REPO}/tree/fc-{issued}",
        f"SHA-256 of the forecast file: {sha}",
        "",
        "Outside forecasts, as available on the issue date:",
    ]
    lines += [f"- {s} ({a}): {u}" for s, a, u in zip(srcs.source, srcs.as_of, srcs.url)]
    (out / "post.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}/table.png and post.md")


if __name__ == "__main__":
    main()
