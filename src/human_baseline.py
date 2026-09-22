"""A human baseline: what does a person score on the posts the model is scored on?

The model's 0.822 has no ceiling to be measured against until a person labels
the same posts under the same conditions. Annotators see the post text only --
no community, score, flair or author -- on a random, class-balanced sample, and
are compared with the model's *out-of-fold* predictions on exactly those posts,
so neither side has seen the answer.

"Unsure" is a legitimate answer. It is what a moderator does when they escalate,
and it lines up with the model's own abstention below a confidence threshold:
the fair comparison is the model abstaining on the same fraction of posts.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import has_franchise_token
from .model import RANDOM_STATE, build_pipeline

UNSURE = "unsure"
ANSWER_COLUMNS = ["id", "annotator", "answer", "seconds"]
N_BOOTSTRAP = 2000


def draw_sample(corpus: pd.DataFrame, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A class-balanced random sample, shuffled so labels do not arrive in runs."""
    classes = sorted(corpus["subreddit"].unique())
    per_class = n // len(classes)
    picked = pd.concat(
        corpus[corpus["subreddit"] == label].sample(per_class, random_state=seed)
        for label in classes)
    return picked.sample(frac=1, random_state=seed)[["id"]].reset_index(drop=True)


def model_predictions(corpus: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold label and confidence for every post, indexed by id."""
    X, y = corpus["text"], corpus["subreddit"]
    pipe = build_pipeline()
    proba = cross_val_predict(
        pipe, X, y, method="predict_proba",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    classes = np.unique(y)
    return pd.DataFrame({
        "model": classes[proba.argmax(axis=1)],
        "confidence": proba.max(axis=1),
    }, index=corpus["id"].to_numpy())


def _bootstrap_ci(correct: np.ndarray, seed: int = RANDOM_STATE) -> tuple[float, float]:
    if len(correct) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(correct, size=(N_BOOTSTRAP, len(correct))).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _accuracy(correct: np.ndarray) -> dict:
    low, high = _bootstrap_ci(correct)
    return {"accuracy": float(correct.mean()) if len(correct) else float("nan"),
            "ci95": [low, high], "n": int(len(correct))}


def score_annotator(answers: pd.DataFrame, corpus: pd.DataFrame,
                    predictions: pd.DataFrame) -> dict:
    """One annotator against the truth, and the model on the same posts."""
    truth = corpus.set_index("id")
    rows = answers.set_index("id").join(truth[["subreddit", "text"]]).join(predictions)
    unsure = (rows["answer"] == UNSURE).to_numpy()
    human_right = (rows["answer"] == rows["subreddit"]).to_numpy()
    model_right = (rows["model"] == rows["subreddit"]).to_numpy()

    # Forced choice: an "unsure" is scored as a coin flip, which is what it
    # would be if the annotator had to pick. Comparable to the model's accuracy.
    forced = np.where(unsure, 0.5, human_right.astype(float))

    # Triage comparison: the model abstains on as many posts as the human did,
    # choosing its own least-confident ones, and is scored on the rest.
    n_abstain = int(unsure.sum())
    keep = np.ones(len(rows), dtype=bool)
    if n_abstain:
        keep[np.argsort(rows["confidence"].to_numpy(), kind="stable")[:n_abstain]] = False

    answered = ~unsure
    named = has_franchise_token(rows["text"]).to_numpy()
    return {
        "n_posts": int(len(rows)),
        "unsure_rate": float(unsure.mean()),
        "median_seconds_per_post": float(rows["seconds"].median()),
        "human_forced_choice": _accuracy(forced),
        "model_same_posts": _accuracy(model_right.astype(float)),
        "human_when_answering": _accuracy(human_right[answered].astype(float)),
        "model_at_same_coverage": _accuracy(model_right[keep].astype(float)),
        "model_on_posts_human_answered": _accuracy(model_right[answered].astype(float)),
        "model_on_posts_human_was_unsure": _accuracy(model_right[unsure].astype(float)),
        "by_franchise_word": {
            "with": {"human_forced_choice": _accuracy(forced[named]),
                     "model": _accuracy(model_right[named].astype(float))},
            "without": {"human_forced_choice": _accuracy(forced[~named]),
                        "model": _accuracy(model_right[~named].astype(float))},
        },
        "kappa_human_vs_model": float(cohen_kappa_score(
            rows["answer"][answered], rows["model"][answered]))
            if answered.sum() > 1 else float("nan"),
    }


def agreement(answers: pd.DataFrame) -> dict:
    """Pairwise Cohen's kappa between annotators, "unsure" counted as a label.

    Low agreement between people means the task itself is ambiguous, and no
    model should be expected to exceed it.
    """
    wide = answers.pivot(index="id", columns="annotator", values="answer")
    pairs = {}
    for a, b in combinations(sorted(wide.columns), 2):
        both = wide[[a, b]].dropna()
        if len(both) < 2:
            continue
        pairs[f"{a}~{b}"] = {
            "n": int(len(both)),
            "raw_agreement": float((both[a] == both[b]).mean()),
            "kappa": float(cohen_kappa_score(both[a], both[b])),
        }
    return pairs


def score(answers: pd.DataFrame, corpus: pd.DataFrame,
          predictions: pd.DataFrame) -> dict:
    return {
        "annotators": {
            name: score_annotator(group, corpus, predictions)
            for name, group in answers.groupby("annotator")
        },
        "inter_annotator": agreement(answers),
    }
