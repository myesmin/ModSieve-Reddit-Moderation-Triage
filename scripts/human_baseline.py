"""Label a blind sample of posts by hand, then score people against the model.

    python scripts/human_baseline.py label --annotator mohona   # resumable
    python scripts/human_baseline.py score                      # writes the report

Every annotator labels the same sample, so their answers can be compared with
each other as well as with the model. Label before looking at the model's
predictions, and without looking posts up.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data import load_corpus
from src.human_baseline import (ANSWER_COLUMNS, UNSURE, draw_sample,
                                model_predictions, score)

OUT = ROOT / "reports" / "human_baseline"
SAMPLE = OUT / "sample.csv"
KEYS = {"m": "marvel", "h": "harrypotter", "u": UNSURE}
MAX_CHARS = 1200


def answers_path(annotator: str) -> Path:
    return OUT / f"answers_{annotator}.csv"


def load_sample(corpus: pd.DataFrame, n: int) -> pd.DataFrame:
    """Fixed on first use and committed, so every annotator sees the same posts."""
    if not SAMPLE.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        draw_sample(corpus, n=n).to_csv(SAMPLE, index=False)
    return pd.read_csv(SAMPLE, dtype=str)


def load_answers() -> pd.DataFrame:
    files = sorted(OUT.glob("answers_*.csv"))
    if not files:
        return pd.DataFrame(columns=ANSWER_COLUMNS)
    return pd.concat((pd.read_csv(f, dtype={"id": str}) for f in files),
                     ignore_index=True)


def label(annotator: str, n: int) -> None:
    corpus = load_corpus()
    text = corpus.set_index("id")["text"]
    sample = load_sample(corpus, n)
    path = answers_path(annotator)
    done = set(pd.read_csv(path, dtype=str)["id"]) if path.exists() else set()
    todo = [i for i in sample["id"] if i not in done]
    if not todo:
        print(f"{annotator} has labelled all {len(sample)} posts")
        return

    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(ANSWER_COLUMNS)
        for position, post_id in enumerate(todo, start=len(done) + 1):
            body = text[post_id]
            if len(body) > MAX_CHARS:
                body = body[:MAX_CHARS] + " [...]"
            print("\033[2J\033[H", end="")
            print(f"post {position}/{len(sample)}\n")
            print(textwrap.fill(body, 88) + "\n")
            started = time.monotonic()
            while True:
                key = input("[m] marvel  [h] harrypotter  [u] unsure  [q] quit > ")
                key = key.strip().lower()
                if key in KEYS or key == "q":
                    break
            if key == "q":
                print(f"saved {position - 1}/{len(sample)}; re-run to continue")
                return
            writer.writerow([post_id, annotator, KEYS[key],
                             round(time.monotonic() - started, 1)])
            handle.flush()
    print(f"done: {len(sample)} posts labelled by {annotator}")


def report() -> None:
    answers = load_answers()
    if answers.empty:
        sys.exit("no answers yet -- run the label command first")
    corpus = load_corpus()
    results = score(answers, corpus, model_predictions(corpus))
    out = OUT / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}\n")

    def fmt(block):
        low, high = block["ci95"]
        return f"{block['accuracy']:.3f} [{low:.3f}, {high:.3f}]  n={block['n']}"

    for name, r in results["annotators"].items():
        print(f"{name}: {r['n_posts']} posts, {r['unsure_rate']:.1%} unsure, "
              f"{r['median_seconds_per_post']:.1f}s median per post")
        print(f"  human, forced choice     {fmt(r['human_forced_choice'])}")
        print(f"  model, same posts        {fmt(r['model_same_posts'])}")
        print(f"  human, when answering    {fmt(r['human_when_answering'])}")
        print(f"  model, same abstention   {fmt(r['model_at_same_coverage'])}")
        print(f"  model where human unsure {fmt(r['model_on_posts_human_was_unsure'])}")
        print(f"  kappa human vs model     {r['kappa_human_vs_model']:.3f}\n")
    for pair, a in results["inter_annotator"].items():
        print(f"{pair}: agreement {a['raw_agreement']:.1%}, kappa {a['kappa']:.3f} "
              f"(n={a['n']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    labelling = sub.add_parser("label")
    labelling.add_argument("--annotator", required=True)
    labelling.add_argument("--n", type=int, default=200,
                           help="sample size; only used the first time")
    sub.add_parser("score")
    args = parser.parse_args()
    if args.command == "label":
        label(args.annotator, args.n)
    else:
        report()


if __name__ == "__main__":
    main()
