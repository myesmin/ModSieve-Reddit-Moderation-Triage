"""Every headline number in the README is produced here, not asserted."""
from __future__ import annotations

import json
import os
import pickle
import tempfile
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     learning_curve)

from .data import (has_franchise_token, load_corpus, strip_franchise_tokens)
from .model import RANDOM_STATE, build_pipeline, keyword_rule
from .policy import best_operating_point, flag_precision, sweep

PRECISION_FLOORS = (0.95, 0.97, 0.99)
BASE_RATES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10)

# A single 5-fold split puts the accuracy estimate at the mercy of row order:
# concatenating marvel-first rather than harrypotter-first moves it by ~1.3
# points, which is larger than most of the differences we care about. Every
# headline number is therefore averaged over repeated splits and reported with
# its spread.
N_REPEATS = 5
SEEDS = tuple(range(N_REPEATS))


def _cv():
    return StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)


def _repeated_cv():
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS,
                                   random_state=RANDOM_STATE)


def run(corpus=None) -> dict:
    corpus = load_corpus() if corpus is None else corpus
    X, y = corpus["text"], corpus["subreddit"]
    cv = _cv()
    results: dict = {}

    results["dataset"] = {
        "n_posts": int(len(corpus)),
        "class_balance": y.value_counts().to_dict(),
        "empty_body_rate": float((corpus["content"].fillna("").str.strip() == "").mean()),
        "median_title_words": float(corpus["title"].fillna("").str.split().str.len().median()),
        "no_franchise_token_rate": float((~has_franchise_token(X)).mean()),
    }

    # --- baselines ------------------------------------------------------
    majority = float(y.value_counts(normalize=True).max())
    results["baselines"] = {
        "majority_class": majority,
        "keyword_regex": float(accuracy_score(y, keyword_rule(X))),
    }

    # --- the model ------------------------------------------------------
    pipe = build_pipeline()
    repeated = _repeated_cv()
    acc = cross_val_score(pipe, X, y, cv=repeated, scoring="accuracy")
    f1 = cross_val_score(pipe, X, y, cv=repeated, scoring="f1_macro")
    results["model"] = {
        "cv_accuracy_mean": float(acc.mean()),
        "cv_accuracy_std": float(acc.std()),
        "cv_macro_f1_mean": float(f1.mean()),
        "lift_over_keyword": float(acc.mean() - results["baselines"]["keyword_regex"]),
    }

    # Ablation: how much of the signal is just proper nouns?
    ablated = cross_val_score(build_pipeline(), strip_franchise_tokens(X), y,
                              cv=_repeated_cv())
    results["model"]["cv_accuracy_franchise_words_removed"] = float(ablated.mean())
    results["model"]["ablation_drop"] = float(acc.mean() - ablated.mean())

    # --- the north star: coverage at a precision floor -------------------
    classes = np.unique(y)
    # Out-of-fold probabilities under several independent fold assignments, so
    # the operating point is reported with its variability rather than as a
    # single lucky split.
    probas = [
        cross_val_predict(pipe, X, y, method="predict_proba",
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        for seed in SEEDS
    ]
    results["operating_points"] = {}
    for floor in PRECISION_FLOORS:
        points = [best_operating_point(y, pr, classes, floor) for pr in probas]
        found = [p for p in points if p is not None]
        if not found:
            results["operating_points"][f"precision_{floor}"] = None
            continue
        coverages = [p.coverage for p in found]
        summary = found[0].as_dict()
        summary.update({
            "coverage": float(np.mean(coverages)),
            "coverage_std": float(np.std(coverages)),
            "coverage_min": float(np.min(coverages)),
            "coverage_max": float(np.max(coverages)),
            "precision": float(np.mean([p.precision for p in found])),
            "threshold": float(np.median([p.threshold for p in found])),
            "n_seeds_reaching_floor": len(found),
            "n_seeds": len(SEEDS),
        })
        results["operating_points"][f"precision_{floor}"] = summary
    proba = probas[0]
    results["threshold_sweep"] = [p.as_dict() for p in sweep(y, proba, classes, 0.99)]

    # --- the base-rate correction ---------------------------------------
    at99 = results["operating_points"]["precision_0.99"]
    p = at99["precision"] if at99 else float(acc.mean())
    results["flag_precision_by_base_rate"] = {
        str(r): flag_precision(p, r) for r in BASE_RATES
    }

    # --- does more data help? --------------------------------------------
    sizes, _, test = learning_curve(
        build_pipeline(min_df=1), X, y, cv=cv,
        train_sizes=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
        scoring="accuracy", shuffle=True, random_state=0)
    results["learning_curve"] = {
        "train_sizes": [int(s) for s in sizes],
        "cv_accuracy": [float(s) for s in test.mean(axis=1)],
    }

    # --- what does it cost to run? ---------------------------------------
    fitted = build_pipeline().fit(X, y)
    sample = [X.iloc[0]]
    fitted.predict_proba(sample)                       # warm up
    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        fitted.predict_proba(sample)
        latencies.append((time.perf_counter() - t0) * 1000)

    batch = (list(X) * 5)[:10_000]
    t0 = time.perf_counter()
    fitted.predict_proba(batch)
    batch_seconds = time.perf_counter() - t0

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as handle:
        pickle.dump(fitted, handle)
        size_kb = os.path.getsize(handle.name) / 1024
    os.unlink(handle.name)

    results["serving"] = {
        "model_size_kb": float(size_kb),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "throughput_docs_per_sec": float(len(batch) / batch_seconds),
        "cpu_hours_per_million_docs": float(
            batch_seconds / len(batch) * 1_000_000 / 3600),
    }
    return results


def main() -> None:
    from pathlib import Path
    results = run()
    out = Path(__file__).resolve().parent.parent / "reports" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    d, m, b = results["dataset"], results["model"], results["baselines"]
    print(f"\n  posts                {d['n_posts']:,} ({d['empty_body_rate']:.1%} empty bodies)")
    print(f"  majority baseline    {b['majority_class']:.3f}")
    print(f"  keyword regex        {b['keyword_regex']:.3f}")
    print(f"  model (5-fold CV)    {m['cv_accuracy_mean']:.3f}  (+{m['lift_over_keyword']:.3f} over regex)")
    for floor, point in results["operating_points"].items():
        if point:
            print(f"  {floor}: coverage {point['coverage']:.1%} "
                  f"at precision {point['precision']:.3f} (threshold {point['threshold']})")
        else:
            print(f"  {floor}: UNREACHABLE at any threshold")


if __name__ == "__main__":
    main()
