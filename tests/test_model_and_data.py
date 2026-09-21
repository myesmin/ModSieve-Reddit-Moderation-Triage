"""Corpus invariants and the baseline the model has to beat."""
import pandas as pd
import pytest

from src.data import has_franchise_token, load_corpus, strip_franchise_tokens
from src.model import build_pipeline, keyword_rule


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_corpus_is_balanced_and_complete(corpus):
    assert len(corpus) == 2_000
    assert corpus["subreddit"].value_counts().to_dict() == {
        "harrypotter": 1_000, "marvel": 1_000}
    assert not corpus["id"].duplicated().any()


def test_corpus_is_title_only_in_practice(corpus):
    """The finding the whole project rests on -- guard it against silent change."""
    empty = (corpus["content"].fillna("").str.strip() == "").mean()
    assert empty > 0.95, f"bodies are no longer mostly empty ({empty:.1%})"


def test_strip_franchise_tokens_removes_names():
    text = pd.Series(["Thanos fights the Avengers", "Snape at Hogwarts"])
    stripped = strip_franchise_tokens(text)
    for word in ("Thanos", "Avengers", "Snape", "Hogwarts"):
        assert word.lower() not in " ".join(stripped).lower()


def test_has_franchise_token_detects_both_sides():
    text = pd.Series(["Hogwarts castle", "Iron Man suit", "my cat is cute"])
    assert has_franchise_token(text).tolist() == [True, True, False]


def test_keyword_rule_classifies_the_obvious_cases():
    text = pd.Series(["Thanos snaps", "Hermione at Hogwarts"])
    assert keyword_rule(text).tolist() == ["marvel", "harrypotter"]


def test_keyword_rule_falls_back_when_ambiguous():
    text = pd.Series(["a lovely afternoon", "Thanos meets Hermione"])
    assert keyword_rule(text, fallback="harrypotter").tolist() == [
        "harrypotter", "harrypotter"]


def test_pipeline_vectorises_inside_the_estimator():
    """Guards the leak that was in the original notebook."""
    steps = dict(build_pipeline().named_steps)
    assert "tfidf" in steps and "clf" in steps


def test_model_beats_the_keyword_baseline(corpus):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score

    X, y = corpus["text"], corpus["subreddit"]
    baseline = accuracy_score(y, keyword_rule(X))
    model = cross_val_score(build_pipeline(), X, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean()
    assert model > baseline, f"model {model:.3f} did not beat regex {baseline:.3f}"
