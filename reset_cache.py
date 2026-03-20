#!/usr/bin/env python3
"""
reset_cache.py
==============
Run this ONCE before your first live pipeline run to clear any
stale synthetic data that was generated in demo mode.

What it deletes:
  cache/scores/score_matrix.parquet  — synthetic LLM scores (MUST delete)
  cache/fred_data.parquet            — synthetic FRED data  (MUST delete)

What it keeps:
  cache/texts/**/*.txt               — any real text files already scraped

Usage:
  python reset_cache.py              # dry run (shows what would be deleted)
  python reset_cache.py --confirm    # actually deletes
"""

import sys
import argparse
from pathlib import Path

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "cache"

TO_DELETE = [
    CACHE_DIR / "scores" / "score_matrix.parquet",
    CACHE_DIR / "fred_data.parquet",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="Actually delete files (default: dry run)")
    args = parser.parse_args()

    print("Cache reset utility")
    print("=" * 50)

    for path in TO_DELETE:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            if args.confirm:
                path.unlink()
                print(f"  DELETED  {path.relative_to(ROOT)}  ({size_kb:.1f} KB)")
            else:
                print(f"  WOULD DELETE  {path.relative_to(ROOT)}  ({size_kb:.1f} KB)")
        else:
            print(f"  NOT FOUND  {path.relative_to(ROOT)}")

    # Report text cache status
    txt_files = list((CACHE_DIR / "texts").rglob("*.txt"))
    print(f"\n  Keeping {len(txt_files)} cached text files in cache/texts/")

    if not args.confirm:
        print("\nDry run complete. Add --confirm to actually delete.")
    else:
        print("\nDone. Re-run pipeline with: python src/pipeline.py --live")


if __name__ == "__main__":
    main()
