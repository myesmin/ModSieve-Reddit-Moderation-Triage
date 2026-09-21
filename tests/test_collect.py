"""The collector's job is to survive a long run. These test that it does."""
import pytest

from src.collect import (Checkpoint, CollectionConfig, RateLimiter, RawComment,
                         RawPost, collect, credential_problems, flatten,
                         load_collected)


def make_post(post_id, subreddit="marvel", comments=()):
    return RawPost(
        id=post_id, subreddit=subreddit, title=f"title {post_id}", body="",
        score=10, num_comments=len(comments), created_utc=1_700_000_000.0,
        upvote_ratio=0.95, flair=None, is_self=False, permalink=f"/r/x/{post_id}",
        comments=tuple(RawComment(body=c, score=1) for c in comments),
    )


class FakeSource:
    """Deterministic stand-in for PRAW; records what was asked of it."""

    def __init__(self, posts_by_sub):
        self.posts_by_sub = posts_by_sub
        self.calls = []

    def iter_posts(self, subreddit, limit, comments_per_post):
        self.calls.append((subreddit, limit, comments_per_post))
        yield from self.posts_by_sub.get(subreddit, [])[:limit]


# --- flatten ---------------------------------------------------------------

def test_flatten_keeps_comments_raw_and_structured():
    import json
    post = make_post("a1", comments=["first comment", "second comment"])
    row = flatten(post)
    comments = json.loads(row["comments_json"])
    assert row["n_comments_collected"] == 2
    assert [c["body"] for c in comments] == ["first comment", "second comment"]
    # The metadata prepare.py needs to strip bots and moderator stickies.
    assert {"author", "is_stickied", "distinguished"} <= set(comments[0])


def test_flatten_does_not_filter_or_assemble_text():
    """Collection stays raw; cleaning decisions belong to prepare.py."""
    row = flatten(make_post("a1"))
    assert "text" not in row
    assert row["comments_json"] == "[]"


# --- rate limiting ---------------------------------------------------------

def test_rate_limiter_paces_calls():
    now, slept = [0.0], []

    def clock():
        return now[0]

    def sleeper(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(60, clock=clock, sleeper=sleeper)
    limiter.acquire()        # first call is free
    limiter.acquire()        # second must wait out the 1s interval
    assert slept and slept[0] == pytest.approx(1.0)


def test_rate_limiter_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        RateLimiter(0)


# --- checkpointing and resume ---------------------------------------------

def test_checkpoint_roundtrips(tmp_path):
    path = tmp_path / "cp.json"
    cp = Checkpoint.load(path)
    cp.add("abc")
    cp.save()
    assert "abc" in Checkpoint.load(path)


def test_rerun_collects_nothing_new(tmp_path):
    source = FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]})
    config = CollectionConfig(
        subreddits=("marvel",), posts_per_subreddit=10, batch_size=4,
        output_dir=tmp_path / "raw", checkpoint_path=tmp_path / "raw/.cp.json")
    limiter = RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None)

    first = collect(source, config, limiter=limiter)
    assert first["marvel"] == 10

    # Same config, fresh checkpoint load -- resume must skip everything.
    second = collect(source, config, limiter=limiter)
    assert second["marvel"] == 0, "resume re-collected posts it already had"


def test_partial_run_resumes_from_checkpoint(tmp_path):
    config = CollectionConfig(
        subreddits=("marvel",), posts_per_subreddit=10, batch_size=4,
        output_dir=tmp_path / "raw", checkpoint_path=tmp_path / "raw/.cp.json")
    limiter = RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None)

    collect(FakeSource({"marvel": [make_post(f"p{i}") for i in range(4)]}),
            config, limiter=limiter)
    resumed = collect(
        FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]}),
        config, limiter=limiter)
    assert resumed["marvel"] == 6      # only the 6 it had not seen


def test_batches_are_written_incrementally(tmp_path):
    source = FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]})
    config = CollectionConfig(
        subreddits=("marvel",), posts_per_subreddit=10, batch_size=4,
        output_dir=tmp_path / "raw", checkpoint_path=tmp_path / "raw/.cp.json")
    collect(source, config,
            limiter=RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None))

    parts = sorted((tmp_path / "raw/subreddit=marvel").glob("part-*.parquet"))
    assert len(parts) == 3                       # 4 + 4 + 2
    assert len(load_collected(tmp_path / "raw")) == 10


def test_multiple_subreddits_are_kept_separate(tmp_path):
    source = FakeSource({
        "marvel": [make_post(f"m{i}", "marvel") for i in range(3)],
        "lotr": [make_post(f"l{i}", "lotr") for i in range(5)],
    })
    config = CollectionConfig(
        subreddits=("marvel", "lotr"), posts_per_subreddit=10, batch_size=100,
        output_dir=tmp_path / "raw", checkpoint_path=tmp_path / "raw/.cp.json")
    written = collect(source, config,
                      limiter=RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None))

    assert written == {"marvel": 3, "lotr": 5}
    frame = load_collected(tmp_path / "raw")
    assert set(frame["subreddit"]) == {"marvel", "lotr"}


def test_load_collected_errors_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_collected(tmp_path / "nothing")


# --- credentials -----------------------------------------------------------

def test_credentials_pass_when_real_values_present():
    assert credential_problems({"REDDIT_CLIENT_ID": "abc123",
                                "REDDIT_CLIENT_SECRET": "def456"}) == []


def test_credentials_flag_missing_keys():
    problems = credential_problems({})
    assert len(problems) == 2 and all("not set" in p for p in problems)


def test_credentials_flag_example_placeholders():
    problems = credential_problems({"REDDIT_CLIENT_ID": "your_client_id_here",
                                    "REDDIT_CLIENT_SECRET": "your_client_secret_here"})
    assert len(problems) == 2 and all("placeholder" in p for p in problems)
