"""collect -> Parquet -> prepare, end to end, with realistic comment noise."""
import pandas as pd

from src.collect import (CollectionConfig, RateLimiter, RawComment, RawPost,
                         collect, load_collected)
from src.prepare import PrepConfig, prepare


class Source:
    def iter_posts(self, subreddit, limit, comments_per_post, skip=None):
        for i in range(limit):
            yield RawPost(
                id=f"{subreddit}{i}", subreddit=subreddit,
                title=f"discussion thread {i} about {subreddit} things", body="",
                score=1, num_comments=3, created_utc=1_600_000_000 + i * 60,
                upvote_ratio=0.9, flair=None, is_self=True, permalink="",
                comments=(
                    RawComment(f"Welcome to r/{subreddit}! Read the rules.", 1,
                               author="AutoModerator", is_stickied=True,
                               distinguished="moderator"),
                    RawComment(f"saw this on r/{subreddit} yesterday, love this sub",
                               5, author="fan_account"),
                    RawComment("genuinely great point", 3, author="another_fan"),
                ),
            )


def test_collected_data_prepares_without_leakage(tmp_path):
    config = CollectionConfig(
        subreddits=("marvel", "harrypotter"), posts_per_subreddit=20,
        comments_per_post=10, batch_size=7,          # forces several Parquet parts
        output_dir=tmp_path / "raw", checkpoint_path=tmp_path / "raw/.cp.json")
    collect(Source(), config,
            limiter=RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None))

    prepared = prepare(load_collected(tmp_path / "raw"),
                       PrepConfig(include_comments=True))
    report = prepared.report
    text = " ".join(prepared.train["text"].tolist() + prepared.test["text"].tolist())

    assert report["input_posts"] == 40
    assert report["cleaning"]["comments"]["kept"] == 80        # 2 fans x 40 posts
    assert report["cleaning"]["comments"]["bot"] == 40         # every AutoModerator
    assert "Welcome to" not in text
    assert "r/marvel" not in text and "r/harrypotter" not in text
    assert "this sub" not in text
    assert "genuinely great point" in text                     # real signal survives


def test_default_preparation_excludes_comments_and_engagement(tmp_path):
    """The data contract: only what exists when a post is submitted."""
    config = CollectionConfig(
        subreddits=("marvel", "harrypotter"), posts_per_subreddit=20,
        comments_per_post=10, output_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "raw/.cp.json")
    collect(Source(), config,
            limiter=RateLimiter(60, clock=lambda: 0.0, sleeper=lambda s: None))

    prepared = prepare(load_collected(tmp_path / "raw"))
    everything = pd.concat([prepared.train, prepared.test])
    text = " ".join(everything["text"])

    assert "genuinely great point" not in text      # comments stay out
    for column in ("score", "num_comments", "upvote_ratio", "comments_json"):
        assert column not in everything.columns
    assert prepared.report["cleaning"]["comments"] == "excluded by data contract"
