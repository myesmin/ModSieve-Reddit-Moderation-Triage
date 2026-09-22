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
    # When this record was taken. Everything above describes the post *as of*
    # this moment -- engagement fields keep changing afterwards -- and the gap
    # from created_utc says how long moderators had already had to act on it.
    snapshot_utc: float = 0.0
    domain: str = ""
    over_18: bool = False
    spoiler: bool = False


class PostSource(Protocol):
    """What the collector needs from a Reddit client.

    `skip(post_id)` is checked before a post's comments are fetched. Comments
    cost one request per post; the listing that yields ids costs one request
    per hundred. Skipping at the id stage is what makes resuming cheap.
    """

    def iter_posts(self, subreddit: str, limit: int, comments_per_post: int,
                   skip: Callable[[str], bool] | None = None) -> Iterator[RawPost]:
        ...


LISTINGS = ("new", "top")


@dataclass
class CollectionConfig:
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS
    listing: str = "new"
    # Reddit's listings stop at ~1,000 posts, so a larger target does nothing
    # for `new`; repeat runs over days are how the corpus grows.
    posts_per_subreddit: int = 1_000
    # Off by default: at submission time a post has no comments, so a model
    # that learns from them cannot run where the decision is made.
    comments_per_post: int = 0
    requests_per_minute: int = 60
    batch_size: int = 250
    log_every: int = 25
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


def _log_progress(subreddit: str, count: int, target: int, started: float) -> None:
    elapsed = time.monotonic() - started
    per_minute = count / elapsed * 60 if elapsed > 0 else 0.0
    remaining = max(target - count, 0)
    eta = f"~{remaining / per_minute:.0f} min left" if per_minute else "eta unknown"
    logger.info("r/%s: %d/%d new posts, %.0f/min, %s",
                subreddit, count, target, per_minute, eta)


def _next_part(output_dir: Path, subreddit: str) -> int:
    """First unused part number, so a resumed run appends instead of overwriting."""
    existing = list((Path(output_dir) / f"subreddit={subreddit}").glob("part-*.parquet"))
    if not existing:
        return 0
    return max(int(p.stem.split("-")[1]) for p in existing) + 1


def collect(source: PostSource, config: CollectionConfig,
            limiter: RateLimiter | None = None,
            checkpoint: Checkpoint | None = None) -> dict[str, int]:
    """Collect every configured subreddit. Returns new posts written per sub."""
    limiter = limiter or RateLimiter(config.requests_per_minute)
    checkpoint = checkpoint or Checkpoint.load(config.checkpoint_path)
    written: dict[str, int] = {}

    for subreddit in config.subreddits:
        batch: list[dict] = []
        part = _next_part(config.output_dir, subreddit)
        count = 0
        started = time.monotonic()
        logger.info("r/%s: starting, target %d posts (%d already collected overall)",
                    subreddit, config.posts_per_subreddit, len(checkpoint.seen))

        posts = source.iter_posts(subreddit, config.posts_per_subreddit,
                                  config.comments_per_post,
                                  skip=checkpoint.__contains__)
        for post in posts:
            if post.id in checkpoint:
                continue            # safety net for sources that ignore `skip`
            batch.append(flatten(post))
            checkpoint.add(post.id)
            count += 1
            if config.log_every and count % config.log_every == 0:
                _log_progress(subreddit, count, config.posts_per_subreddit, started)
            if len(batch) >= config.batch_size:
                write_batch(batch, config.output_dir, subreddit, part)
                checkpoint.save()             # checkpoint only after a durable write
                batch, part = [], part + 1
            if config.comments_per_post:
                # Only comment fetches cost a request per post. Listings cost one
                # per hundred posts and PRAW paces those itself; waiting here
                # without comments would turn an 8,000-post pass from minutes
                # into over two hours of sleeping.
                limiter.acquire()
        if batch:
            write_batch(batch, config.output_dir, subreddit, part)
            checkpoint.save()
        written[subreddit] = count
        logger.info("r/%s: done, %d new posts in %.1f min", subreddit, count,
                    (time.monotonic() - started) / 60)

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

    def __init__(self, reddit=None, listing: str = "new") -> None:
        if listing not in LISTINGS:
            raise ValueError(f"listing must be one of {LISTINGS}, got {listing!r}")
        self._listing = listing
        if reddit is None:
            import os
            import praw
            reddit = praw.Reddit(
                client_id=os.environ["REDDIT_CLIENT_ID"],
                client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                user_agent=os.environ.get(
                    "REDDIT_USER_AGENT", "ModSieve/1.0"),
            )
        self._reddit = reddit

    def iter_posts(self, subreddit: str, limit: int, comments_per_post: int,
                   skip: Callable[[str], bool] | None = None) -> Iterator[RawPost]:
        sub = self._reddit.subreddit(subreddit)
        skip = skip or (lambda _id: False)
        considered: set[str] = set()
        # For `top`, widening windows reach further back than any single call,
        # but they overlap: an all-time top post is usually a top post of the
        # year too. Each id is counted once, and already-collected ids count
        # toward the target without their comments being fetched again.
        for submission in self._listing_iter(sub):
            if len(considered) >= limit:
                return
            if submission.id in considered:
                continue
            considered.add(submission.id)
            if skip(submission.id):
                continue
            yield self._to_post(submission, subreddit, comments_per_post)

    def lookup(self, ids: list[str]):
        """Current state of posts by id, including ones hidden from listings."""
        return self._reddit.info(fullnames=[f"t3_{i}" for i in ids])

    def _listing_iter(self, sub):
        if self._listing == "new":
            yield from sub.new(limit=None)
            return
        for time_filter in ("all", "year", "month", "week"):
            yield from sub.top(limit=None, time_filter=time_filter)

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
            snapshot_utc=time.time(),
            domain=getattr(submission, "domain", "") or "",
            over_18=bool(getattr(submission, "over_18", False)),
            spoiler=bool(getattr(submission, "spoiler", False)),
        )
