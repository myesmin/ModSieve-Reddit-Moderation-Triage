"""Scalable collection: many communities, with comment threads.

The existing corpus is title-only (97.7% of bodies are empty), which is the
measured reason coverage stalls. Comments are the fix: they carry 10-100x the
text per post. Collecting them means tens of thousands of API calls, so this
module is built for runs that are long enough to fail partway through:

* **resumable** - every post id that has been written is checkpointed, so a
  re-run skips work instead of duplicating it
* **rate limited** - requests are paced rather than fired as fast as the
  network allows
* **incremental** - batches are flushed to disk as they fill, so memory stays
  flat whether the target is 2,000 posts or 200,000

The Reddit client is injected rather than imported, so the collection logic is
testable without network access or credentials.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = (
    "harrypotter", "marvel", "DC_Cinematic", "StarWars",
    "lotr", "StarTrek", "anime", "comicbooks",
)


@dataclass(frozen=True)
class RawComment:
    body: str
    score: int
    author: str = ""
    is_stickied: bool = False
    distinguished: str | None = None     # "moderator" / "admin" / None


@dataclass(frozen=True)
class RawPost:
    """Provider-agnostic post. A source adapter produces these."""
    id: str
    subreddit: str
    title: str
    body: str
    score: int
    num_comments: int
    created_utc: float
    upvote_ratio: float
    flair: str | None
    is_self: bool
    permalink: str
    comments: tuple[RawComment, ...] = ()


class PostSource(Protocol):
    """What the collector needs from a Reddit client."""

    def iter_posts(self, subreddit: str, limit: int,
                   comments_per_post: int) -> Iterator[RawPost]:
        ...


@dataclass
class CollectionConfig:
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS
    posts_per_subreddit: int = 2_000
    comments_per_post: int = 20
    requests_per_minute: int = 60
    batch_size: int = 250
    output_dir: Path = Path("Data/raw")
    checkpoint_path: Path = Path("Data/raw/.checkpoint.json")


class RateLimiter:
    """Paces calls to `requests_per_minute`.

    The clock and sleep function are injectable so tests can assert on pacing
    without actually waiting.
    """

    def __init__(self, requests_per_minute: int,
                 clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleeper = sleeper
        self._next_allowed: float | None = None

    def acquire(self) -> None:
        now = self._clock()
        if self._next_allowed is not None and now < self._next_allowed:
            self._sleeper(self._next_allowed - now)
            now = max(now, self._next_allowed)
        self._next_allowed = now + self._interval


@dataclass
class Checkpoint:
    """Post ids already written, so an interrupted run resumes cleanly."""
    path: Path
    seen: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        path = Path(path)
        if path.exists():
            payload = json.loads(path.read_text())
            return cls(path=path, seen=set(payload.get("seen", [])))
        return cls(path=path)

    def add(self, post_id: str) -> None:
        self.seen.add(post_id)

    def __contains__(self, post_id: str) -> bool:
        return post_id in self.seen

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"seen": sorted(self.seen)}))


def flatten(post: RawPost) -> dict:
    """One post -> one row, kept raw.

    Comments are stored as a JSON string rather than a nested Parquet column:
    batches whose comments happen to be all-empty would otherwise infer a
    different schema from batches that have them, and the parts would stop
    concatenating cleanly. Filtering and text assembly happen in prepare.py, so
    the raw data stays complete and every cleaning decision stays reversible.
    """
    record = asdict(post)
    comments = record.pop("comments")
    record["comments_json"] = json.dumps(comments, ensure_ascii=False)
    record["n_comments_collected"] = len(comments)
    return record


def write_batch(rows: list[dict], output_dir: Path, subreddit: str,
                part: int) -> Path:
    target = Path(output_dir) / f"subreddit={subreddit}"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"part-{part:05d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def collect(source: PostSource, config: CollectionConfig,
            limiter: RateLimiter | None = None,
            checkpoint: Checkpoint | None = None) -> dict[str, int]:
    """Collect every configured subreddit. Returns new posts written per sub."""
    limiter = limiter or RateLimiter(config.requests_per_minute)
    checkpoint = checkpoint or Checkpoint.load(config.checkpoint_path)
    written: dict[str, int] = {}

    for subreddit in config.subreddits:
        batch: list[dict] = []
        part = 0
        count = 0
        posts = source.iter_posts(subreddit, config.posts_per_subreddit,
                                  config.comments_per_post)
        for post in posts:
            limiter.acquire()
            if post.id in checkpoint:
                continue                      # already have it; resume cheaply
            batch.append(flatten(post))
            checkpoint.add(post.id)
            count += 1
            if len(batch) >= config.batch_size:
                write_batch(batch, config.output_dir, subreddit, part)
                checkpoint.save()             # checkpoint only after a durable write
                batch, part = [], part + 1
        if batch:
            write_batch(batch, config.output_dir, subreddit, part)
            checkpoint.save()
        written[subreddit] = count
        logger.info("collected %d new posts from r/%s", count, subreddit)

    return written


def load_collected(output_dir: Path) -> pd.DataFrame:
    """Read every parquet part back into one frame."""
    parts = sorted(Path(output_dir).glob("subreddit=*/part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no collected data under {output_dir}")
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


REQUIRED_CREDENTIALS = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")


def credential_problems(env) -> list[str]:
    """Name every missing or placeholder credential, before any request is made.

    PRAW is lazy: a bad secret surfaces only as a bare 401 on the first request.
    Checking up front turns that into a message that says what to fix.
    """
    problems = []
    for key in REQUIRED_CREDENTIALS:
        value = (env.get(key) or "").strip()
        if not value:
            problems.append(f"{key} is not set")
        elif value.lower().startswith("your_"):
            problems.append(f"{key} still holds the .env.example placeholder")
    return problems


class PrawSource:
    """Adapter over PRAW. Imported lazily so tests never need the dependency."""

    def __init__(self, reddit=None) -> None:
        if reddit is None:
            import os
            import praw
            reddit = praw.Reddit(
                client_id=os.environ["REDDIT_CLIENT_ID"],
                client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                user_agent=os.environ.get(
                    "REDDIT_USER_AGENT", "Reddit-NLP-HP-Marvel/1.0"),
            )
        self._reddit = reddit

    def iter_posts(self, subreddit: str, limit: int,
                   comments_per_post: int) -> Iterator[RawPost]:
        sub = self._reddit.subreddit(subreddit)
        seen = 0
        # `top` over widening windows reaches further back than any single call.
        for time_filter in ("all", "year", "month", "week"):
            if seen >= limit:
                return
            for submission in sub.top(limit=None, time_filter=time_filter):
                if seen >= limit:
                    return
                yield self._to_post(submission, subreddit, comments_per_post)
                seen += 1

    @staticmethod
    def _to_post(submission, subreddit: str, comments_per_post: int) -> RawPost:
        comments: list[RawComment] = []
        if comments_per_post:
            # `replace_more(limit=0)` drops the "load more comments" stubs
            # rather than spending a request expanding each one. Nothing is
            # filtered here: bots, moderator stickies and deleted comments are
            # removed in prepare.py, where the removals are counted and reported.
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:comments_per_post]:
                author = getattr(comment, "author", None)
                comments.append(RawComment(
                    body=getattr(comment, "body", "") or "",
                    score=getattr(comment, "score", 0),
                    author=author.name if author else "",
                    is_stickied=bool(getattr(comment, "stickied", False)),
                    distinguished=getattr(comment, "distinguished", None),
                ))
        return RawPost(
            id=submission.id,
            subreddit=subreddit,
            title=submission.title or "",
            body=submission.selftext or "",
            score=submission.score,
            num_comments=submission.num_comments,
            created_utc=submission.created_utc,
            upvote_ratio=getattr(submission, "upvote_ratio", float("nan")),
            flair=submission.link_flair_text,
            is_self=submission.is_self,
            permalink=getattr(submission, "permalink", ""),
            comments=tuple(comments),
        )
