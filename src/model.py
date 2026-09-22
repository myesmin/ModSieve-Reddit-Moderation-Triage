"""The model pipeline and the regex baseline."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import FRANCHISE_TOKENS

RANDOM_STATE = 42


def build_pipeline(**overrides) -> Pipeline:
    """TF-IDF + logistic regression.

    Vectorisation lives *inside* the pipeline so it is refit within each CV
    fold. Fitting the vectoriser on the full training set before
    cross-validating leaks test-fold vocabulary and inflates scores.
    """
    params = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    params.update(overrides)
    return Pipeline([
        ("tfidf", TfidfVectorizer(**params)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


_MARVEL = re.compile(FRANCHISE_TOKENS["marvel"], re.I)
_POTTER = re.compile(FRANCHISE_TOKENS["harrypotter"], re.I)


def keyword_rule(text: pd.Series, fallback: str = "harrypotter") -> np.ndarray:
    """Hand-written regex baseline that the model should beat."""
    def classify(value: str) -> str:
        is_marvel = bool(_MARVEL.search(value))
        is_potter = bool(_POTTER.search(value))
        if is_marvel and not is_potter:
            return "marvel"
        if is_potter and not is_marvel:
            return "harrypotter"
        return fallback

    return text.map(classify).to_numpy()
