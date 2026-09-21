"""CLI for corpus collection.

    python scripts/collect.py --subreddits marvel harrypotter --posts 2000 --comments 20

Safe to interrupt and re-run: completed posts are checkpointed and skipped.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# `python scripts/collect.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.collect import (DEFAULT_SUBREDDITS, Checkpoint, CollectionConfig,
                         PrawSource, RateLimiter, collect, credential_problems)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subreddits", nargs="+", default=list(DEFAULT_SUBREDDITS))
    parser.add_argument("--listing", choices=["new", "top"], default="new",
                        help="new: the posts a moderation queue sees (default); "
                             "top: all-time highlights, mostly image posts")
    parser.add_argument("--posts", type=int, default=1_000,
                        help="posts per subreddit (Reddit listings stop near 1,000)")
    parser.add_argument("--comments", type=int, default=0,
                        help="comments per post; off by default because they do "
                             "not exist when a post is submitted")
    parser.add_argument("--rpm", type=int, default=60,
                        help="requests per minute (be kind to the API)")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to Data/raw/<listing>, so samples never mix")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    args.output = args.output or ROOT / "Data" / "raw" / args.listing

    load_dotenv(ROOT / ".env")
    problems = credential_problems(os.environ)
    if problems:
        sys.exit("Cannot collect -- fix .env first:\n  - " + "\n  - ".join(problems)
                 + "\n\nCreate a 'script' app at https://www.reddit.com/prefs/apps "
                   "and paste its id and secret into .env.")

    config = CollectionConfig(
        subreddits=tuple(args.subreddits),
        listing=args.listing,
        posts_per_subreddit=args.posts,
        comments_per_post=args.comments,
        requests_per_minute=args.rpm,
        output_dir=args.output,
        checkpoint_path=args.output / ".checkpoint.json",
    )
    checkpoint = Checkpoint.load(config.checkpoint_path)
    if checkpoint.seen:
        logging.info("resuming: %d posts already collected", len(checkpoint.seen))

    try:
        written = collect(PrawSource(listing=config.listing), config,
                          limiter=RateLimiter(config.requests_per_minute),
                          checkpoint=checkpoint)
    except Exception as exc:
        # prawcore raises ResponseException(401) for a wrong id/secret pair.
        if "401" in str(exc):
            sys.exit("Reddit rejected the credentials (HTTP 401). Check that "
                     "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env belong "
                     "to the same app, and that the app type is 'script'.")
        raise

    total = sum(written.values())
    logging.info("done: %d new posts across %d subreddits", total, len(written))
    for subreddit, count in written.items():
        logging.info("  r/%-16s %6d", subreddit, count)


if __name__ == "__main__":
    main()
