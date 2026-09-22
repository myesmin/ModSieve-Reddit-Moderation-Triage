"""Delayed moderation labels.

Reddit hides removed posts from its listings, so a removal can only be observed
by recording a post while it is live (src/collect.py) and looking it up again
by id once moderators have had time to act. This module is the second half.

Two things make the label trustworthy:

* **Removal is not deletion.** A moderator removing a post is a moderation
  decision; an author deleting their own post is not. They get different
  outcomes, and only the former is a positive label.
* **Only posts seen young are unbiased.** A post first recorded at five days
  old had already survived five days of moderation. `hours_old_at_snapshot`
  travels with every label so analysis can keep only posts caught early.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

LABEL_AFTER_HOURS = 48
LOOKUP_BATCH = 100          # reddit.info accepts up to 100 ids per request

KEPT = "kept"
REMOVED = "removed_by_moderation"
DELETED = "deleted_by_author"
MISSING = "missing"

# removed_by_category values that mean someone other than the author acted.
MODERATION_CATEGORIES = frozenset({
    "moderator", "automod_filtered", "reddit", "anti_evil_ops",
    "content_takedown", "copyright_takedown", "community_ops",
})
AUTHOR_CATEGORIES = frozenset({"deleted", "author"})


def outcome(removed_by_category: str | None) -> str:
    if removed_by_category in MODERATION_CATEGORIES:
        return REMOVED
    if removed_by_category in AUTHOR_CATEGORIES:
        return DELETED
    if removed_by_category is None:
        return KEPT
    # An unfamiliar category is surfaced rather than silently guessed at.
    return f"other:{removed_by_category}"


def load_labels(labels_dir: Path) -> pd.DataFrame:
    parts = sorted(Path(labels_dir).glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=["id"])
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def due_for_labelling(raw: pd.DataFrame, labelled: set[str], now: float,
                      after_hours: float = LABEL_AFTER_HOURS) -> pd.DataFrame:
    """Posts old enough to label that have not been labelled yet."""
    old_enough = raw["created_utc"] <= now - after_hours * 3600
    return raw[old_enough & ~raw["id"].isin(labelled)]


def _next_part(labels_dir: Path) -> int:
    existing = list(Path(labels_dir).glob("part-*.parquet"))
    return max((int(p.stem.split("-")[1]) for p in existing), default=-1) + 1


def recheck(lookup: Callable[[list[str]], Iterable], raw: pd.DataFrame,
            labels_dir: Path, now: float | None = None,
            after_hours: float = LABEL_AFTER_HOURS) -> dict:
    """Label every due post; append results; return a summary.

    `lookup(ids)` returns objects with `.id` and `.removed_by_category`. It is
    injected so this is testable without network access.
    """
    now = time.time() if now is None else now
    labels_dir = Path(labels_dir)
    already = set(load_labels(labels_dir)["id"])
    due = due_for_labelling(raw.drop_duplicates("id"), already, now, after_hours)
    if due.empty:
        return {"due": 0, "labelled": 0, "outcomes": {}}

    found: dict[str, str | None] = {}
    ids = due["id"].tolist()
    for start in range(0, len(ids), LOOKUP_BATCH):
        for post in lookup(ids[start:start + LOOKUP_BATCH]):
            found[post.id] = getattr(post, "removed_by_category", None)

    rows = []
    for record in due.itertuples(index=False):
        present = record.id in found
        category = found.get(record.id)
        snapshot = getattr(record, "snapshot_utc", 0.0) or 0.0
        rows.append({
            "id": record.id,
            "subreddit": record.subreddit,
            "checked_utc": now,
            "hours_since_creation": (now - record.created_utc) / 3600,
            # 0.0 snapshot_utc means the record predates timestamping.
            "hours_old_at_snapshot": ((snapshot - record.created_utc) / 3600
                                      if snapshot else float("nan")),
            "removed_by_category": category,
            "outcome": outcome(category) if present else MISSING,
        })

    labels = pd.DataFrame(rows)
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_dir / f"part-{_next_part(labels_dir):05d}.parquet",
                      index=False)
    return {"due": len(due), "labelled": len(labels),
            "outcomes": labels["outcome"].value_counts().to_dict()}


def removal_rate(labels: pd.DataFrame, max_age_at_snapshot: float = 2.0) -> dict:
    """Removal rate among posts caught young enough to be unbiased."""
    young = labels[labels["hours_old_at_snapshot"] <= max_age_at_snapshot]
    eligible = young[young["outcome"].isin([KEPT, REMOVED])]
    n, k = len(eligible), int((eligible["outcome"] == REMOVED).sum())
    return {"eligible_posts": n, "removed": k,
            "rate": k / n if n else float("nan"),
            "by_subreddit": eligible.groupby("subreddit")["outcome"]
                                    .apply(lambda s: float((s == REMOVED).mean()))
                                    .round(4).to_dict()}
