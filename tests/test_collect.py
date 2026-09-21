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
    """Deterministic stand-in for PRAW. `fetched` records every post whose
    comments would have cost a real request."""

    def __init__(self, posts_by_sub):
        self.posts_by_sub = posts_by_sub
        self.fetched = []

    def iter_posts(self, subreddit, limit, comments_per_post, skip=None):
        skip = skip or (lambda _id: False)
        for post in self.posts_by_sub.get(subreddit, [])[:limit]:
            if skip(post.id):
                continue
            self.fetched.append(post.id)
            yield post


def no_wait():
    return RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None)


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


# --- resume must be cheap and must not lose data --------------------------

def resume_config(tmp_path, batch_size=4):
    return CollectionConfig(
        subreddits=("marvel",), posts_per_subreddit=10, batch_size=batch_size,
        output_dir=tmp_path / "raw", checkpoint_path=tmp_path / "raw/.cp.json")


def test_resume_does_not_refetch_collected_posts(tmp_path):
    config = resume_config(tmp_path)
    collect(FakeSource({"marvel": [make_post(f"p{i}") for i in range(4)]}),
            config, limiter=no_wait())

    second = FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]})
    collect(second, config, limiter=no_wait())
    assert second.fetched == [f"p{i}" for i in range(4, 10)], \
        "already-collected posts cost a comment request on resume"


def test_resume_preserves_data_from_the_earlier_run(tmp_path):
    """Regression: part numbering restarted at 0 and overwrote the first run."""
    config = resume_config(tmp_path)
    collect(FakeSource({"marvel": [make_post(f"p{i}") for i in range(4)]}),
            config, limiter=no_wait())
    collect(FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]}),
            config, limiter=no_wait())

    on_disk = load_collected(tmp_path / "raw")
    assert sorted(on_disk["id"]) == sorted(f"p{i}" for i in range(10))
    assert not on_disk["id"].duplicated().any()


def test_rate_limiter_is_only_charged_for_real_requests(tmp_path):
    config = resume_config(tmp_path)
    collect(FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]}),
            config, limiter=no_wait())

    acquired = []
    counting = RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None)
    counting.acquire = lambda: acquired.append(1)
    collect(FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]}),
            config, limiter=counting)
    assert acquired == [], "a fully-resumed run should make no paced requests"


def test_progress_is_logged(tmp_path, caplog):
    import logging
    config = resume_config(tmp_path, batch_size=100)
    config.log_every = 5
    with caplog.at_level(logging.INFO, logger="src.collect"):
        collect(FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]}),
                config, limiter=no_wait())
    progress = [r.message for r in caplog.records if "new posts," in r.message]
    assert len(progress) == 2 and "5/10" in progress[0] and "10/10" in progress[1]


# --- the PRAW adapter, against a fake Reddit client -------------------------

class _Comments:
    def replace_more(self, limit):
        pass

    def list(self):
        return []


class _Submission:
    def __init__(self, post_id):
        self.id = post_id
        self.title, self.selftext = f"title {post_id}", ""
        self.score, self.num_comments, self.created_utc = 1, 0, 1.0
        self.upvote_ratio, self.link_flair_text = 0.9, None
        self.is_self, self.permalink = True, ""
        self.comments = _Comments()


class _Subreddit:
    """`top` returns overlapping windows, as Reddit's real listings do."""

    def __init__(self, windows):
        self.windows = windows

    def top(self, limit, time_filter):
        return [_Submission(i) for i in self.windows.get(time_filter, [])]

    def new(self, limit):
        return [_Submission(i) for i in self.windows.get("new", [])]


class _Reddit:
    def __init__(self, windows):
        self.windows = windows

    def subreddit(self, name):
        return _Subreddit(self.windows)


def test_praw_source_counts_overlapping_windows_once():
    from src.collect import PrawSource
    windows = {"all": ["a", "b", "c"], "year": ["b", "c", "d", "e"]}
    ids = [p.id for p in PrawSource(_Reddit(windows), listing="top").iter_posts("x", 10, 0)]
    assert ids == ["a", "b", "c", "d", "e"]


def test_praw_source_stops_at_the_limit_across_windows():
    from src.collect import PrawSource
    windows = {"all": ["a", "b"], "year": ["b", "c", "d"]}
    ids = [p.id for p in PrawSource(_Reddit(windows), listing="top").iter_posts("x", 3, 0)]
    assert ids == ["a", "b", "c"]


def test_praw_source_counts_skipped_posts_toward_the_limit():
    """A resumed run should top up to the target, not collect `limit` more."""
    from src.collect import PrawSource
    windows = {"all": ["a", "b", "c", "d", "e"]}
    ids = [p.id for p in PrawSource(_Reddit(windows), listing="top").iter_posts(
        "x", 4, 0, skip=lambda i: i in {"a", "b"})]
    assert ids == ["c", "d"]


def test_praw_source_defaults_to_the_new_listing():
    from src.collect import PrawSource
    windows = {"new": ["n1", "n2"], "all": ["t1"]}
    posts = list(PrawSource(_Reddit(windows)).iter_posts("x", 10, 0))
    assert [p.id for p in posts] == ["n1", "n2"]
    assert all(p.snapshot_utc > 0 for p in posts)     # every record is timestamped


def test_praw_source_rejects_unknown_listing():
    from src.collect import PrawSource
    with pytest.raises(ValueError):
        PrawSource(_Reddit({}), listing="controversial")


def test_collection_defaults_match_the_queue():
    config = CollectionConfig()
    assert config.listing == "new"
    assert config.comments_per_post == 0


def test_no_pacing_when_comments_are_off(tmp_path):
    acquired = []
    limiter = RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None)
    limiter.acquire = lambda: acquired.append(1)
    config = resume_config(tmp_path)
    config.comments_per_post = 0
    collect(FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]}),
            config, limiter=limiter)
    assert acquired == [], "listing-only collection waited once per post"


def test_pacing_applies_when_comments_are_fetched(tmp_path):
    acquired = []
    limiter = RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None)
    limiter.acquire = lambda: acquired.append(1)
    config = resume_config(tmp_path)
    config.comments_per_post = 5
    collect(FakeSource({"marvel": [make_post(f"p{i}") for i in range(10)]}),
            config, limiter=limiter)
    assert len(acquired) == 10
