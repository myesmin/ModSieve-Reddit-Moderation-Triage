"""Delayed labels must separate removal from deletion and never double-count."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.labels import (DELETED, KEPT, MISSING, REMOVED, load_labels, outcome,
                        recheck, removal_rate)

HOUR = 3600
NOW = 1_700_000_000.0


def raw_posts(n=5, created=NOW - 72 * HOUR, age_at_snapshot=0.5):
    return pd.DataFrame([{
        "id": f"p{i}", "subreddit": "marvel", "created_utc": created,
        "snapshot_utc": created + age_at_snapshot * HOUR,
    } for i in range(n)])


class Lookup:
    """Fake reddit.info: returns a category per id, and records batch sizes."""

    def __init__(self, categories, missing=()):
        self.categories, self.missing, self.batches = categories, set(missing), []

    def __call__(self, ids):
        self.batches.append(len(ids))
        return [SimpleNamespace(id=i, removed_by_category=self.categories.get(i))
                for i in ids if i not in self.missing]


@pytest.mark.parametrize("category, expected", [
    (None, KEPT), ("moderator", REMOVED), ("automod_filtered", REMOVED),
    ("reddit", REMOVED), ("deleted", DELETED), ("author", DELETED),
    ("something_new", "other:something_new"),
])
def test_outcome(category, expected):
    assert outcome(category) == expected


def test_author_deletion_is_not_a_removal(tmp_path):
    lookup = Lookup({"p0": "moderator", "p1": "deleted"})
    recheck(lookup, raw_posts(3), tmp_path, now=NOW)
    labels = load_labels(tmp_path).set_index("id")["outcome"]
    assert labels.to_dict() == {"p0": REMOVED, "p1": DELETED, "p2": KEPT}


def test_posts_are_not_labelled_before_moderators_have_had_time(tmp_path):
    fresh = raw_posts(3, created=NOW - 10 * HOUR)
    summary = recheck(Lookup({}), fresh, tmp_path, now=NOW, after_hours=48)
    assert summary["due"] == 0 and load_labels(tmp_path).empty


def test_each_post_is_labelled_once(tmp_path):
    raw = raw_posts(4)
    recheck(Lookup({}), raw, tmp_path, now=NOW)
    second = recheck(Lookup({}), raw, tmp_path, now=NOW + HOUR)
    assert second["labelled"] == 0
    assert len(load_labels(tmp_path)) == 4


def test_new_labels_append_rather_than_overwrite(tmp_path):
    recheck(Lookup({}), raw_posts(2), tmp_path, now=NOW)
    more = pd.concat([raw_posts(2), raw_posts(5).iloc[2:]])
    recheck(Lookup({}), more, tmp_path, now=NOW + HOUR)
    assert sorted(load_labels(tmp_path)["id"]) == [f"p{i}" for i in range(5)]


def test_ids_the_api_does_not_return_are_marked_missing(tmp_path):
    recheck(Lookup({}, missing={"p1"}), raw_posts(3), tmp_path, now=NOW)
    labels = load_labels(tmp_path).set_index("id")["outcome"]
    assert labels["p1"] == MISSING and labels["p0"] == KEPT


def test_lookups_are_batched_at_100(tmp_path):
    lookup = Lookup({})
    recheck(lookup, raw_posts(250), tmp_path, now=NOW)
    assert lookup.batches == [100, 100, 50]


def test_removal_rate_uses_only_posts_seen_young(tmp_path):
    young = raw_posts(4, age_at_snapshot=0.5)                    # 1 of 4 removed
    old = raw_posts(4, age_at_snapshot=100).assign(               # survivors only
        id=lambda d: "old" + d["id"])
    lookup = Lookup({"p0": "moderator"})
    recheck(lookup, pd.concat([young, old]), tmp_path, now=NOW)

    rate = removal_rate(load_labels(tmp_path), max_age_at_snapshot=2)
    assert rate["eligible_posts"] == 4 and rate["removed"] == 1
    assert rate["rate"] == pytest.approx(0.25)


def test_removal_rate_excludes_author_deletions_from_the_denominator(tmp_path):
    recheck(Lookup({"p0": "moderator", "p1": "deleted"}), raw_posts(4), tmp_path, now=NOW)
    rate = removal_rate(load_labels(tmp_path))
    assert rate["eligible_posts"] == 3 and rate["removed"] == 1
