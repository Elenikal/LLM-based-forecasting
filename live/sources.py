"""
live/sources.py
===============
Real-time inputs for the live forecast record.

  - FOMC minutes 2025+        (reuses src/data/text_pull.py; calendars page)
  - Beige Books 2024+         (new site layout: one summary page + 12 district
                               pages per issue, stitched back into the full
                               report so inputs match the pre-2024 cache)
  - FRED quarterly panel      (ragged edge: a quarter enters only once it is
                               complete for that series)
"""

import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import BVAR_VARS, FRED_API_KEY
from data.fred_pull import apply_transformations
from data.text_pull import (BASE, BEIGE_DIR, FOMC_DIR, date_to_quarter,
                            download_fomc_minutes, html_to_text, safe_get)

# District order of the pre-2024 single-page report
DISTRICTS = ["boston", "new-york", "philadelphia", "cleveland", "richmond",
             "atlanta", "chicago", "st-louis", "minneapolis", "kansas-city",
             "dallas", "san-francisco"]

BB_INDEX_PAGES = ["/monetarypolicy/beigebook{year}.htm",
                  "/monetarypolicy/publications/beige-book-default.htm"]


# ─────────────────────────────────────────────────────────────────────────────
# Text
# ─────────────────────────────────────────────────────────────────────────────

def _beige_book_codes(year: int) -> list[str]:
    codes = set()
    for page in BB_INDEX_PAGES:
        r = safe_get(BASE + page.format(year=year), retries=1)
        if r is not None:
            codes |= set(re.findall(rf"beigebook({year}\d\d)-summary\.htm", r.text))
    return sorted(codes)


def _release_date(summary_html: str) -> str:
    """Publication date: the line holding only a date, e.g. 'June 04, 2025'."""
    m = re.search(r"^\s*([A-Z][a-z]+ \d{1,2}, \d{4})\s*$", summary_html, re.M)
    return datetime.strptime(m.group(1), "%B %d, %Y").strftime("%Y-%m-%d")


def _article_text(html: str) -> str:
    art = BeautifulSoup(html, "html.parser").find(id="article")
    return html_to_text(str(art)) if art else html_to_text(html)


def download_beige_books_new(year_start: int = 2024,
                             year_end: int | None = None) -> list[dict]:
    year_end = year_end or date.today().year
    docs = []
    for year in range(year_start, year_end + 1):
        for code in _beige_book_codes(year):
            existing = list(BEIGE_DIR.glob(f"*_{code}.txt"))
            if existing:
                path = existing[0]
            else:
                r = safe_get(f"{BASE}/monetarypolicy/beigebook{code}-summary.htm")
                if r is None:
                    continue
                rel = _release_date(r.text)
                parts = [html_to_text(r.text)]
                for d in DISTRICTS:
                    rd = safe_get(f"{BASE}/monetarypolicy/beigebook{code}-{d}.htm")
                    if rd is not None:
                        parts.append(_article_text(rd.text))
                # file name keeps the release date and the Fed's issue code
                path = BEIGE_DIR / f"{date_to_quarter(rel)}_{rel}_{code}.txt"
                path.write_text("\n\n".join(parts), encoding="utf-8")
                print(f"  ✓ Beige Book {rel} ({len(parts) - 1} districts)")
            q, rel = path.stem.split("_")[:2]
            text = path.read_text(encoding="utf-8")
            docs.append({"date": rel, "quarter": q, "source": "beige_book",
                         "text": text, "n_words": len(text.split())})
    return docs


def load_new_texts(fomc_from: int = 2025, beige_from: int = 2024) -> list[dict]:
    """All documents released after the frozen paper sample."""
    fomc = download_fomc_minutes(fomc_from, date.today().year)
    beige = download_beige_books_new(beige_from)
    return sorted(fomc + beige, key=lambda d: d["date"])


# ─────────────────────────────────────────────────────────────────────────────
# FRED (ragged edge)
# ─────────────────────────────────────────────────────────────────────────────

def _to_quarterly(s: pd.Series, today: date) -> pd.Series:
    """Quarter value = last observation in the quarter (as in fred_pull), kept
    only if the quarter is complete for this series."""
    s = s.dropna()
    if len(s) == 0:
        return s
    gaps = s.index.to_series().diff().dt.days.median()
    if gaps > 80:                                    # quarterly
        q = s.copy()
        q.index = q.index.to_period("Q")
        return q
    q = s.resample("QE").last()
    q.index = q.index.to_period("Q")
    if gaps > 20:                                    # monthly: need 3rd month
        last_month = s.index[-1].to_period("M")
        if last_month != q.index[-1].asfreq("M", how="end"):
            q = q.iloc[:-1]
    elif pd.Timestamp(today) <= q.index[-1].end_time:  # daily: quarter over
        q = q.iloc[:-1]
    return q


def load_fred_live(today: date | None = None) -> tuple[pd.DataFrame, dict]:
    from fredapi import Fred
    today = today or date.today()
    fred = Fred(api_key=FRED_API_KEY)
    raw, last_obs = {}, {}
    for sid in BVAR_VARS:
        s = fred.get_series(sid, observation_start="1995-01-01")
        last_obs[sid] = str(s.dropna().index[-1].date())
        raw[sid] = _to_quarterly(s, today)
    raw = pd.DataFrame(raw)
    return apply_transformations(raw).loc["1999Q1":], last_obs
