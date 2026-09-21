"""Raw collected posts -> a clean, leakage-checked, split dataset.

Every removal is counted and written to a report, because a cleaning step that
silently drops rows is indistinguishable from a bug. The three things this
stage guards against, in order of how badly they would mislead us:

1. **Label leakage.** Comments carry the community's own fingerprints --
   AutoModerator stickies ("Welcome to r/harrypotter!"), moderator notices,
   users writing "crossposting from r/marvel". A model will happily learn
   those instead of the content, and coverage will rise for the wrong reason.
2. **Contradictory duplicates.** A post crossposted to two communities appears
   twice with two different labels. Kept, it is label noise; split across
   train and test, it is test-set contamination.
3. **Temporal leakage.** A random split trains on posts written after the ones
   it is tested on. Deployment only ever sees the future, so the test set is
   the most recent slice of each community.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

# Accounts whose comments are the platform talking, not the community.
BOT_AUTHORS = frozenset({
    "automoderator", "remindmebot", "sneakpeekbot", "wikitextbot",
    "repostsleuthbot", "savevideo", "stabbot", "linkifybot", "b0trank",
    "converter-bot", "same_subreddit_bot", "totesmessenger",
})
REMOVED_BODIES = frozenset({"", "[deleted]", "[removed]"})

# r/marvel, /r/marvel, R/Marvel -- any community, not just the two labelled ones.
SUBREDDIT_MENTION = re.compile(r"(?<![\w/])/?r/[A-Za-z0-9_]{2,21}\b", re.I)
SELF_REFERENCE = re.compile(r"\bthis\s+(?:sub|subreddit)\b", re.I)
URL = re.compile(r"https?://\S+")
WHITESPACE = re.compile(r"\s+")


@dataclass
class PrepConfig:
    min_words: int = 3
    test_fraction: float = 0.2
    split: str = "temporal"          # "temporal" or "random"
    random_state: int = 42


@dataclass
class PreparedDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    report: dict = field(default_factory=dict)


# --- comment and text cleaning ----------------------------------------------

def is_bot(author: str) -> bool:
    name = (author or "").lower()
    return name in BOT_AUTHORS or name.endswith("bot")


def clean_comments(comments: list[dict], counts: Counter) -> list[str]:
    """Keep only comments written by the community itself."""
    kept = []
    for comment in comments:
        body = (comment.get("body") or "").strip()
        if body in REMOVED_BODIES:
            counts["deleted_or_removed"] += 1
        elif is_bot(comment.get("author", "")):
            counts["bot"] += 1
        elif comment.get("is_stickied") or comment.get("distinguished"):
            counts["moderator_or_stickied"] += 1
        else:
            kept.append(body)
            counts["kept"] += 1
    return kept


def strip_leakage(text: str, counts: Counter) -> str:
    """Remove explicit references to a community, which reveal the label."""
    text, n_mentions = SUBREDDIT_MENTION.subn(" ", text)
    text, n_self = SELF_REFERENCE.subn(" ", text)
    counts["subreddit_mentions"] += n_mentions
    counts["self_references"] += n_self
    return text


def normalise(text: str) -> str:
    return WHITESPACE.sub(" ", URL.sub(" ", text)).strip()


def title_key(title: str) -> str:
    """Duplicate detection key: case, punctuation and spacing do not count."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


# --- the pipeline -------------------------------------------------------------

def assemble(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean comments, strip leakage and build the `text` column."""
    comment_counts: Counter = Counter()
    leak_counts: Counter = Counter()
    texts, n_kept = [], []

    for row in raw.itertuples(index=False):
        comments = json.loads(getattr(row, "comments_json", "[]") or "[]")
        kept = clean_comments(comments, comment_counts)
        parts = [row.title or "", row.body or "", *kept]
        text = strip_leakage(" ".join(p for p in parts if p), leak_counts)
        texts.append(normalise(text))
        n_kept.append(len(kept))

    out = raw.copy()
    out["text"] = texts
    out["n_comments_kept"] = n_kept
    return out, {"comments": dict(comment_counts), "leakage": dict(leak_counts)}


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    key = df["title"].map(title_key)
    labels_per_key = df.groupby(key)["subreddit"].nunique()

    # Same title under more than one community: the label is genuinely
    # ambiguous, so every copy goes -- keeping one would pick a winner at random.
    conflicting = key.map(labels_per_key) > 1
    df, key = df[~conflicting], key[~conflicting]

    # Same title repeated within one community: keep the first.
    repeated = key.duplicated(keep="first")
    return df[~repeated], {
        "cross_posted_conflict": int(conflicting.sum()),
        "duplicate_within_subreddit": int(repeated.sum()),
    }


def split(df: pd.DataFrame, config: PrepConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stratified split: each community contributes the same test fraction."""
    trains, tests, cutoffs = [], [], {}
    for label, group in df.groupby("subreddit"):
        if config.split == "temporal":
            ordered = group.sort_values("created_utc")
        elif config.split == "random":
            ordered = group.sample(frac=1.0, random_state=config.random_state)
        else:
            raise ValueError(f"unknown split method {config.split!r}")
        n_test = max(1, round(len(ordered) * config.test_fraction))
        trains.append(ordered.iloc[:-n_test])
        tests.append(ordered.iloc[-n_test:])
        if config.split == "temporal":
            cutoffs[label] = float(ordered.iloc[-n_test]["created_utc"])
    info = {"method": config.split, "test_fraction": config.test_fraction}
    if cutoffs:
        info["test_starts_utc"] = cutoffs
    return (pd.concat(trains, ignore_index=True),
            pd.concat(tests, ignore_index=True), info)


def data_hash(train: pd.DataFrame, test: pd.DataFrame) -> str:
    """Content fingerprint, so a model can record exactly what it trained on."""
    digest = hashlib.sha256()
    for name, frame in (("train", train), ("test", test)):
        digest.update(name.encode())
        digest.update("\n".join(sorted(frame["id"].astype(str))).encode())
    return digest.hexdigest()[:12]


def prepare(raw: pd.DataFrame, config: PrepConfig | None = None) -> PreparedDataset:
    config = config or PrepConfig()
    report: dict = {"config": asdict(config), "input_posts": int(len(raw))}

    df, report["cleaning"] = assemble(raw)
    df, report["deduplication"] = deduplicate(df)
    train, test, report["split"] = split(df, config)

    # Very short posts ("Neat", "Mood") are noise to learn from, so they leave
    # the training set. They stay in the test set: production still receives
    # them, and escalating them to a human is the correct behaviour. Filtering
    # them from test too would make the test set easier than reality and
    # overstate coverage.
    too_short = train["text"].str.split().str.len() < config.min_words
    report["too_short_removed_from_train"] = int(too_short.sum())
    report["short_posts_kept_in_test"] = int(
        (test["text"].str.split().str.len() < config.min_words).sum())
    train = train[~too_short].reset_index(drop=True)
    report["output_posts"] = int(len(train) + len(test))
    report["class_balance"] = {
        "train": train["subreddit"].value_counts().to_dict(),
        "test": test["subreddit"].value_counts().to_dict(),
    }
    report["mean_words_per_post"] = float(
        pd.concat([train, test])["text"].str.split().str.len().mean())
    report["data_hash"] = data_hash(train, test)
    return PreparedDataset(train=train, test=test, report=report)


def write(prepared: PreparedDataset, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.train.to_parquet(output_dir / "train.parquet", index=False)
    prepared.test.to_parquet(output_dir / "test.parquet", index=False)
    (output_dir / "report.json").write_text(json.dumps(prepared.report, indent=2))
    return output_dir


def from_legacy_csv(data_dir: Path) -> pd.DataFrame:
    """Map the original title-only CSVs onto the collector's raw schema."""
    frames = []
    for filename, label in (("harrypotter.csv", "harrypotter"),
                            ("marvel.csv", "marvel")):
        df = pd.read_csv(Path(data_dir) / filename)
        frames.append(pd.DataFrame({
            "id": df["id"], "subreddit": label,
            "title": df["title"].fillna(""), "body": df["content"].fillna(""),
            "score": df["score"], "num_comments": df["num_comments"],
            "created_utc": df["created_utc"], "upvote_ratio": df["upvote_ratio"],
            "flair": df["subreddit_flair"], "comments_json": "[]",
        }))
    return pd.concat(frames, ignore_index=True)
