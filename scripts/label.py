"""Look collected posts up again after 48h and record moderation outcomes.

    python scripts/label.py            # label everything that is due
    python scripts/label.py --stats    # current removal rate, no API calls
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.collect import PrawSource, credential_problems, load_collected
from src.labels import LABEL_AFTER_HOURS, load_labels, recheck, removal_rate

RAW = ROOT / "Data" / "raw" / "new"
LABELS = ROOT / "Data" / "labels"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--after-hours", type=float, default=LABEL_AFTER_HOURS)
    args = parser.parse_args()

    if not args.stats:
        load_dotenv(ROOT / ".env")
        problems = credential_problems(os.environ)
        if problems:
            sys.exit("Cannot label -- fix .env first: " + "; ".join(problems))
        summary = recheck(PrawSource().lookup, load_collected(RAW), LABELS,
                          after_hours=args.after_hours)
        print(f"labelled {summary['labelled']} posts: {json.dumps(summary['outcomes'])}")

    labels = load_labels(LABELS)
    if labels.empty:
        print("no labels yet -- posts become due 48h after they were created")
        return
    stats = removal_rate(labels)
    print(f"total labelled: {len(labels):,}  outcomes: "
          f"{json.dumps(labels['outcome'].value_counts().to_dict())}")
    print(f"removal rate (posts seen <2h old): {stats['removed']}/{stats['eligible_posts']}"
          f" = {stats['rate']:.2%}")
    print(f"by community: {json.dumps(stats['by_subreddit'])}")


if __name__ == "__main__":
    main()
