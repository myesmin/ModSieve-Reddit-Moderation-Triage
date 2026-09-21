"""CLI for data preparation.

    python scripts/prepare.py                  # from collected data in Data/raw
    python scripts/prepare.py --legacy         # from the original 2,000-post CSVs

Writes train.parquet, test.parquet and report.json. Read the report: it counts
every row and comment that was removed, and why.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.collect import load_collected
from src.prepare import PrepConfig, from_legacy_csv, prepare, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "Data" / "raw")
    parser.add_argument("--legacy", action="store_true",
                        help="use the original CSVs in Data/ instead of Data/raw")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-words", type=int, default=3)
    args = parser.parse_args()

    raw = from_legacy_csv(ROOT / "Data") if args.legacy else load_collected(args.raw)
    config = PrepConfig(min_words=args.min_words, test_fraction=args.test_fraction,
                        split=args.split)
    prepared = prepare(raw, config)

    out = args.out or ROOT / "Data" / "processed" / (
        "legacy" if args.legacy else prepared.report["data_hash"])
    write(prepared, out)

    r = prepared.report
    print(f"input posts         {r['input_posts']:,}")
    print(f"cross-post conflict -{r['deduplication']['cross_posted_conflict']}")
    print(f"within-sub repeats  -{r['deduplication']['duplicate_within_subreddit']}")
    print(f"too short (train)   -{r['too_short_removed_from_train']}   "
          f"(kept in test: {r['short_posts_kept_in_test']})")
    print(f"output posts        {r['output_posts']:,}")
    print(f"comments            {json.dumps(r['cleaning']['comments'])}")
    print(f"leakage stripped    {json.dumps(r['cleaning']['leakage'])}")
    print(f"mean words / post   {r['mean_words_per_post']:.1f}")
    print(f"train / test        {r['class_balance']['train']} / {r['class_balance']['test']}")
    print(f"data hash           {r['data_hash']}")
    print(f"\nwrote {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
