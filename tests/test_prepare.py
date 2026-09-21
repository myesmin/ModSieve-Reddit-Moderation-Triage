"""Preparation must remove leakage and contradictions, and say what it removed."""
import json

import pandas as pd
import pytest

from src.prepare import (PrepConfig, clean_comments, deduplicate, is_bot,
                         prepare, strip_leakage, title_key)
from collections import Counter


def comment(body, author="someone", stickied=False, distinguished=None):
    return {"body": body, "score": 1, "author": author,
            "is_stickied": stickied, "distinguished": distinguished}


def raw_row(post_id, subreddit, title, created, comments=()):
    return {"id": post_id, "subreddit": subreddit, "title": title, "body": "",
            "created_utc": created, "comments_json": json.dumps(list(comments))}


# --- comment cleaning ---------------------------------------------------------

def test_bots_moderators_and_deleted_comments_are_dropped():
    counts = Counter()
    kept = clean_comments([
        comment("great scene"),
        comment("Welcome to the sub! Read the rules.", author="AutoModerator"),
        comment("Pinned: spoiler policy", stickied=True),
        comment("Removed for rule 3", distinguished="moderator"),
        comment("[deleted]"),
    ], counts)
    assert kept == ["great scene"]
    assert counts == Counter(kept=1, bot=1, moderator_or_stickied=2,
                             deleted_or_removed=1)


@pytest.mark.parametrize("author, expected", [
    ("AutoModerator", True), ("RemindMeBot", True), ("some_helper_bot", True),
    ("regular_fan", False), ("", False),
])
def test_is_bot(author, expected):
    assert is_bot(author) is expected


# --- leakage ------------------------------------------------------------------

def test_subreddit_mentions_are_stripped():
    counts = Counter()
    text = strip_leakage("crossposting from r/marvel and /r/HarryPotter", counts)
    assert "r/marvel" not in text.lower() and "harrypotter" not in text.lower()
    assert counts["subreddit_mentions"] == 2


def test_self_references_are_stripped():
    counts = Counter()
    text = strip_leakage("I love this sub, best subreddit ever", counts)
    assert "this sub" not in text.lower()
    assert counts["self_references"] == 1


def test_franchise_words_are_not_treated_as_leakage():
    """'Marvel' as content is legitimate signal; only 'r/marvel' is a label leak."""
    text = strip_leakage("the Marvel movies are great", Counter())
    assert "Marvel" in text


# --- deduplication ------------------------------------------------------------

def test_title_key_ignores_case_and_punctuation():
    assert title_key("Hello, World!") == title_key("hello   world")


def test_crossposted_titles_drop_every_copy():
    df = pd.DataFrame([
        {"id": "1", "subreddit": "marvel", "title": "Look at this!"},
        {"id": "2", "subreddit": "harrypotter", "title": "look at this"},
        {"id": "3", "subreddit": "marvel", "title": "unique post"},
    ])
    kept, counts = deduplicate(df)
    assert kept["id"].tolist() == ["3"]
    assert counts["cross_posted_conflict"] == 2


def test_repeats_within_one_subreddit_keep_the_first():
    df = pd.DataFrame([
        {"id": "1", "subreddit": "marvel", "title": "same title"},
        {"id": "2", "subreddit": "marvel", "title": "Same Title"},
    ])
    kept, counts = deduplicate(df)
    assert kept["id"].tolist() == ["1"]
    assert counts["duplicate_within_subreddit"] == 1


# --- end to end ---------------------------------------------------------------

@pytest.fixture
def raw():
    rows = []
    for sub in ("marvel", "harrypotter"):
        for i in range(10):
            rows.append(raw_row(f"{sub}{i}", sub, f"{sub} post number {i} here",
                                created=1_600_000_000 + i * 1000,
                                comments=[comment(f"reply {i} in r/{sub}"),
                                          comment("rules", author="AutoModerator")]))
    return pd.DataFrame(rows)


def test_temporal_split_tests_only_on_the_future(raw):
    prepared = prepare(raw, PrepConfig(test_fraction=0.2, split="temporal"))
    for sub in ("marvel", "harrypotter"):
        train = prepared.train[prepared.train["subreddit"] == sub]
        test = prepared.test[prepared.test["subreddit"] == sub]
        assert train["created_utc"].max() < test["created_utc"].min()


def test_split_is_stratified(raw):
    prepared = prepare(raw, PrepConfig(test_fraction=0.2))
    assert prepared.report["class_balance"]["test"] == {"marvel": 2, "harrypotter": 2}


def test_no_post_lands_in_both_train_and_test(raw):
    prepared = prepare(raw)
    assert not set(prepared.train["id"]) & set(prepared.test["id"])


def test_prepared_text_contains_no_label_leakage(raw):
    prepared = prepare(raw, PrepConfig(include_comments=True))
    text = " ".join(pd.concat([prepared.train, prepared.test])["text"]).lower()
    assert "r/marvel" not in text and "r/harrypotter" not in text
    assert "rules" not in text                    # the AutoModerator comment


def test_report_accounts_for_every_removal(raw):
    report = prepare(raw, PrepConfig(include_comments=True)).report
    assert report["cleaning"]["comments"]["bot"] == 20
    assert report["cleaning"]["leakage"]["subreddit_mentions"] == 20
    assert report["output_posts"] == 20
    assert len(report["data_hash"]) == 12


def test_data_hash_is_deterministic(raw):
    assert prepare(raw).report["data_hash"] == prepare(raw).report["data_hash"]


def test_unknown_split_method_is_rejected(raw):
    with pytest.raises(ValueError):
        prepare(raw, PrepConfig(split="alphabetical"))


def test_short_posts_leave_train_but_stay_in_test():
    """Test must reflect production, which still receives one-word posts."""
    # Short titles differ per community: an identical title in both would be
    # removed earlier as a cross-post conflict, never reaching this rule.
    short = {"marvel": ("Neat", "Wow"), "harrypotter": ("Mood", "Oof")}
    rows = []
    for sub in ("marvel", "harrypotter"):
        newest, oldest = short[sub]
        for i in range(10):
            # the newest post in each community is short, so it lands in test
            title = newest if i == 9 else f"a longer {sub} title {i}"
            rows.append(raw_row(f"{sub}{i}", sub, title, created=1_600_000_000 + i))
        rows.append(raw_row(f"{sub}old", sub, oldest, created=1_500_000_000))
    prepared = prepare(pd.DataFrame(rows), PrepConfig(min_words=3, test_fraction=0.2))

    train_text = prepared.train["text"].tolist()
    test_text = prepared.test["text"].tolist()
    assert "Wow" not in train_text and "Oof" not in train_text
    assert "Neat" in test_text and "Mood" in test_text
    assert prepared.report["too_short_removed_from_train"] == 2
    assert prepared.report["short_posts_kept_in_test"] == 2


def test_output_carries_only_contract_columns(raw):
    raw = raw.assign(score=100, num_comments=50, upvote_ratio=0.9, domain="i.redd.it")
    prepared = prepare(raw)
    columns = set(prepared.train.columns)
    assert {"score", "num_comments", "upvote_ratio", "comments_json"}.isdisjoint(columns)
    assert {"id", "subreddit", "text", "domain"} <= columns
    assert "score" in prepared.report["data_contract"]["dropped_columns"]
