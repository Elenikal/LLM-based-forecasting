"""
data/text_pull.py
=================
Downloads FOMC minutes and Beige Books from federalreserve.gov.

Verified URL patterns (as of March 2026):
  FOMC historical (2007-2020): /monetarypolicy/fomchistorical{YEAR}.htm
  FOMC recent    (2021+):      /monetarypolicy/fomccalendars.htm
  Beige Book index:            /monetarypolicy/beigebook{YEAR}.htm   ← no subdirectory
  Beige Book individual:       /monetarypolicy/beigebook{YYYYMM}.htm
"""

import re
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEXT_DIR

FOMC_DIR  = TEXT_DIR / "fomc_minutes"
BEIGE_DIR = TEXT_DIR / "beige_books"
for d in [FOMC_DIR, BEIGE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASE    = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "AcademicResearch/1.0 (macroeconomics forecasting)"}
SLEEP   = 1.2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def date_to_quarter(date_str: str) -> str:
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"


def safe_get(url: str, retries: int = 2) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            time.sleep(SLEEP)
            return r
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ✗ {url}  ({e})")
                return None
            time.sleep(SLEEP * (attempt + 1))
    return None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def full_url(href: str) -> str:
    return f"{BASE}{href}" if href.startswith("/") else href


# ─────────────────────────────────────────────────────────────────────────────
# FOMC Minutes
# ─────────────────────────────────────────────────────────────────────────────

def _extract_minutes_links(html: str) -> list[dict]:
    soup    = BeautifulSoup(html, "html.parser")
    results = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        m    = re.search(r"fomcminutes(\d{8})\.htm", href)
        if m:
            d        = m.group(1)
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            results.append({
                "date":    date_str,
                "quarter": date_to_quarter(date_str),
                "url":     full_url(href),
                "source":  "fomc_minutes",
            })
    return results


def download_fomc_minutes(year_start: int = 2007,
                          year_end:   int = 2024) -> list[dict]:
    documents     = []
    calendars_raw = None   # fetched once for 2021+

    for year in range(max(year_start, 2007), year_end + 1):
        if year <= 2020:
            url = f"{BASE}/monetarypolicy/fomchistorical{year}.htm"
            r   = safe_get(url)
            index = _extract_minutes_links(r.text) if r else []
        else:
            # Recent meetings live on the calendars page
            if calendars_raw is None:
                r = safe_get(f"{BASE}/monetarypolicy/fomccalendars.htm")
                calendars_raw = r.text if r else ""
            index = [l for l in _extract_minutes_links(calendars_raw)
                     if l["date"].startswith(str(year))]

        if not index:
            print(f"  [FOMC] No minutes found for {year}")

        for item in index:
            cache_path = FOMC_DIR / f"{item['quarter']}_{item['date']}.txt"
            if cache_path.exists():
                text = cache_path.read_text(encoding="utf-8")
            else:
                r = safe_get(item["url"])
                if r is None:
                    continue
                text = html_to_text(r.text)
                for marker in ["Participants", "Staff Review", "Discussion"]:
                    idx = text.find(marker)
                    if idx > 0:
                        text = text[idx:]
                        break
                cache_path.write_text(text, encoding="utf-8")
                print(f"  ✓ FOMC minutes {item['date']} → {item['quarter']}")

            doc           = dict(item)
            doc["text"]   = text
            doc["n_words"]= len(text.split())
            documents.append(doc)

    print(f"  FOMC minutes: {len(documents)} documents loaded.")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Beige Books
# ─────────────────────────────────────────────────────────────────────────────

def get_beige_book_index(year: int) -> list[dict]:
    """
    Correct URL: /monetarypolicy/beigebook{YEAR}.htm  (no subdirectory).
    Individual books link to: /monetarypolicy/beigebook{YYYYMM}.htm
    """
    url = f"{BASE}/monetarypolicy/beigebook{year}.htm"
    r   = safe_get(url)
    if r is None:
        print(f"  [Beige Book] Index unavailable for {year}")
        return []

    soup    = BeautifulSoup(r.text, "html.parser")
    results = []
    seen    = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        m    = re.search(r"beigebook(\d{6,8})\.htm", href)
        if m and "historical" not in href and "archive" not in href:
            raw = m.group(1)
            if len(raw) == 6:
                date_str = f"{raw[:4]}-{raw[4:6]}-01"
            else:
                date_str = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"

            if not date_str.startswith(str(year)):
                continue

            u = full_url(href)
            if u not in seen:
                seen.add(u)
                results.append({
                    "date":    date_str,
                    "quarter": date_to_quarter(date_str),
                    "url":     u,
                    "source":  "beige_book",
                })

    return results


def download_beige_books(year_start: int = 2007,
                         year_end:   int = 2024) -> list[dict]:
    documents = []
    for year in range(year_start, year_end + 1):
        index = get_beige_book_index(year)
        for item in index:
            cache_path = BEIGE_DIR / f"{item['quarter']}_{item['date']}.txt"
            if cache_path.exists():
                text = cache_path.read_text(encoding="utf-8")
            else:
                r = safe_get(item["url"])
                if r is None:
                    continue
                text = html_to_text(r.text)
                cache_path.write_text(text, encoding="utf-8")
                print(f"  ✓ Beige Book {item['date']} → {item['quarter']}")

            doc           = dict(item)
            doc["text"]   = text
            doc["n_words"]= len(text.split())
            documents.append(doc)

    print(f"  Beige Books: {len(documents)} documents loaded.")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Quarter grouping + master loader
# ─────────────────────────────────────────────────────────────────────────────

def group_by_quarter(documents: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for doc in documents:
        q = doc["quarter"]
        if q not in grouped:
            grouped[q] = {"fomc_minutes": [], "beige_book": []}
        grouped[q][doc["source"]].append(doc["text"])
    return grouped


def load_all_texts(year_start: int = 2007, year_end: int = 2024,
                   use_cache: bool = True) -> dict[str, dict]:
    n_cached = len(list(TEXT_DIR.rglob("*.txt")))
    if use_cache and n_cached > 10:
        print(f"  Found {n_cached} cached text files — loading from disk.")

    effective_start = max(year_start, 2007)
    fomc_docs  = download_fomc_minutes(effective_start, year_end)
    beige_docs = download_beige_books(effective_start, year_end)

    by_quarter = group_by_quarter(fomc_docs + beige_docs)
    print(f"\n  Total quarters with text: {len(by_quarter)}")
    return by_quarter


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_texts(year_start: int = 2000,
                              year_end:   int = 2024) -> dict[str, dict]:
    import random
    import pandas as pd
    random.seed(42)
    templates = [
        "Economic activity continued to expand at a moderate pace. Labor market "
        "conditions remained solid. Inflation was running slightly above the 2 "
        "percent objective. Credit conditions were broadly accommodative.",
        "The staff revised up its projection for real GDP growth. Consumer "
        "spending expanded at a robust rate. Wages were rising but inflationary "
        "pressures remained contained.",
        "Economic activity slowed amid heightened uncertainty. Labor market "
        "conditions softened. Inflation declined but remained above target. "
        "Participants expressed concern about downside risks to growth.",
        "Supply chain disruptions weighed on production. Consumer spending was "
        "subdued. Elevated uncertainty about the policy path ahead. Inflation "
        "risks seen as skewed to the upside.",
    ]
    quarters = pd.period_range(f"{year_start}Q1", f"{year_end}Q4", freq="Q")
    return {
        str(q): {
            "fomc_minutes": [random.choice(templates), random.choice(templates)],
            "beige_book":   [random.choice(templates)],
        }
        for q in quarters
    }


if __name__ == "__main__":
    print("Testing Beige Book index URLs:")
    for year in [2010, 2015, 2020, 2022, 2024]:
        idx = get_beige_book_index(year)
        print(f"  {year}: {len(idx)} entries")
    print("\nTesting FOMC minutes:")
    for year in [2010, 2019, 2022, 2024]:
        idx = get_fomc_minutes_index(year) if year <= 2020 else []
        print(f"  {year}: via {'historical' if year <= 2020 else 'calendars'} page")