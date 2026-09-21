"""The decision layer.

A model emits a probability. A *system* has to decide what to do with it.
This module is that gap: it turns scores into one of two actions -- decide
automatically, or escalate to a human -- and reports the only number the
project is really optimising:

    coverage at a fixed precision floor
      = what fraction of the queue can we resolve without human review,
        while staying at or above the precision we promised?
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

AUTO = "auto_resolve"
ESCALATE = "escalate_to_human"


@dataclass(frozen=True)
class OperatingPoint:
    """A threshold, and what it buys."""
    threshold: float
    coverage: float           # fraction of volume handled without a human
    precision: float          # accuracy on the covered slice
    n_covered: int
    n_errors_covered: int
    precision_floor: float
    meets_floor: bool

    def as_dict(self) -> dict:
        return asdict(self)


def decide(confidence: float, threshold: float) -> str:
    return AUTO if confidence >= threshold else ESCALATE


def sweep(y_true, proba, classes, floor: float, grid=None) -> list[OperatingPoint]:
    """Evaluate every candidate threshold."""
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    confidence = proba.max(axis=1)
    predicted = classes[proba.argmax(axis=1)]
    correct = predicted == y_true

    grid = np.arange(0.50, 1.00, 0.01) if grid is None else np.asarray(grid)
    points = []
    for t in grid:
        covered = confidence >= t
        n = int(covered.sum())
        if n == 0:
            continue
        precision = float(correct[covered].mean())
        points.append(OperatingPoint(
            threshold=round(float(t), 4),
            coverage=n / len(y_true),
            precision=precision,
            n_covered=n,
            n_errors_covered=int((~correct[covered]).sum()),
            precision_floor=floor,
            meets_floor=precision >= floor,
        ))
    return points


def best_operating_point(y_true, proba, classes, floor: float) -> OperatingPoint | None:
    """Lowest threshold meeting the precision floor -- i.e. maximum coverage.

    Returns None when no threshold reaches the floor, which is a real and
    reportable outcome, not an error.
    """
    viable = [p for p in sweep(y_true, proba, classes, floor) if p.meets_floor]
    return max(viable, key=lambda p: p.coverage) if viable else None


def flag_precision(classification_precision: float, base_rate: float) -> float:
    """Precision of an *off-topic flag*, which is not classification precision.

    A flag fires when the model confidently disagrees with the subreddit a post
    was actually submitted to. Genuinely off-topic posts are rare, so most
    disagreements are the model being wrong rather than the post being wrong:

        P(truly off-topic | flagged)
          = r*p / (r*p + (1-r)*(1-p))

    where r is the off-topic base rate and p the classification precision.
    Ignoring this turns 99% classification precision into a ~53% flag precision
    claim that will not survive contact with a reviewer.
    """
    true_catch = base_rate * classification_precision
    model_error = (1 - base_rate) * (1 - classification_precision)
    denominator = true_catch + model_error
    return float(true_catch / denominator) if denominator else 0.0
