"""The human baseline must compare people and model on identical posts, fairly."""
import numpy as np
import pandas as pd
import pytest

from src.human_baseline import UNSURE, agreement, draw_sample, score_annotator


def corpus(n=20):
    return pd.DataFrame({
        "id": [f"p{i}" for i in range(n)],
        "subreddit": ["marvel", "harrypotter"] * (n // 2),
        "text": ["thanos again" if i % 2 == 0 else "a quiet evening" for i in range(n)],
    })


def test_sample_is_balanced_shuffled_and_reproducible():
    posts = corpus(100)
    sample = draw_sample(posts, n=20, seed=3)
    labels = posts.set_index("id").loc[sample["id"], "subreddit"]
    assert labels.value_counts().to_dict() == {"marvel": 10, "harrypotter": 10}
    assert sample.equals(draw_sample(posts, n=20, seed=3))
    assert list(sample.columns) == ["id"]          # the label is never shown


def test_scoring_with_unsure_answers():
    posts = corpus(4)
    truth = posts["subreddit"].tolist()            # m, h, m, h
    answers = pd.DataFrame({
        "id": posts["id"], "annotator": "a",
        "answer": [truth[0], truth[1], "harrypotter", UNSURE],
        "seconds": [2.0, 4.0, 6.0, 8.0],
    })
    predictions = pd.DataFrame({
        "model": [truth[0], "marvel", truth[2], truth[3]],
        "confidence": [0.9, 0.95, 0.6, 0.55],
    }, index=posts["id"].to_numpy())

    r = score_annotator(answers, posts, predictions)
    assert r["unsure_rate"] == 0.25
    assert r["median_seconds_per_post"] == 5.0
    # 1 + 1 + 0 + 0.5 (coin flip for unsure) over 4 posts
    assert r["human_forced_choice"]["accuracy"] == pytest.approx(0.625)
    assert r["human_when_answering"]["accuracy"] == pytest.approx(2 / 3)
    assert r["model_same_posts"]["accuracy"] == 0.75
    # The model drops its single least-confident post (p3, which it had right).
    assert r["model_at_same_coverage"]["n"] == 3
    assert r["model_at_same_coverage"]["accuracy"] == pytest.approx(2 / 3)
    assert r["model_on_posts_human_was_unsure"]["accuracy"] == 1.0
    low, high = r["human_forced_choice"]["ci95"]
    assert 0 <= low <= 0.625 <= high <= 1


def test_agreement_counts_unsure_and_skips_unshared_posts():
    answers = pd.DataFrame({
        "id": ["p0", "p1", "p2", "p0", "p1", "p3"],
        "annotator": ["a", "a", "a", "b", "b", "b"],
        "answer": ["marvel", UNSURE, "marvel", "marvel", "harrypotter", "marvel"],
    })
    pair = agreement(answers)["a~b"]
    assert pair["n"] == 2
    assert pair["raw_agreement"] == 0.5
    assert np.isfinite(pair["kappa"])
