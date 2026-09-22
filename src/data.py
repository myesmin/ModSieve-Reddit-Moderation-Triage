"""Loading and assembling the labelled corpus."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

SUBREDDITS = {"harrypotter.csv": "harrypotter", "marvel.csv": "marvel"}

# Tokens that name a franchise outright. Used only for the keyword baseline and
# the ablation, not as model features.
FRANCHISE_TOKENS = {
    "marvel": r"marvel|avenger|thanos|mcu|spider|iron ?man|deadpool|stan lee|"
              r"captain america|thor|loki|hulk|wanda|x-?men",
    "harrypotter": r"potter|hogwarts|hermione|voldemort|snape|weasley|dumbledore|"
                   r"gryffindor|slytherin|hufflepuff|ravenclaw|wizarding|dobby|hagrid",
}
ANY_FRANCHISE_TOKEN = re.compile("|".join(FRANCHISE_TOKENS.values()), re.I)


def load_corpus(data_dir: Path | None = None) -> pd.DataFrame:
    """Return one row per post with `text` and `subreddit` columns.

    `text` is title + body. Note that ~97% of bodies are empty: these are
    predominantly image/link posts, so in practice this is a title-only corpus.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    frames = []
    for filename, label in SUBREDDITS.items():
        df = pd.read_csv(data_dir / filename)
        df["subreddit"] = label
        frames.append(df)

    corpus = pd.concat(frames, ignore_index=True)
    corpus["text"] = (
        corpus["title"].fillna("") + " " + corpus["content"].fillna("")
    ).str.strip()
    return corpus


def strip_franchise_tokens(text: pd.Series) -> pd.Series:
    """Remove every explicit franchise name, for the ablation experiment."""
    return text.str.replace(ANY_FRANCHISE_TOKEN, " ", regex=True)


def has_franchise_token(text: pd.Series) -> pd.Series:
    return text.str.contains(ANY_FRANCHISE_TOKEN, na=False)
