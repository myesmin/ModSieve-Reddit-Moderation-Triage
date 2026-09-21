"""The decision layer is where a probability becomes an action."""
import numpy as np
import pytest

from src.policy import (AUTO, ESCALATE, best_operating_point, decide,
                        flag_precision, sweep)

CLASSES = np.array(["a", "b"])


def proba_from(confidences, correct):
    """Build a 2-class probability matrix with given confidence and correctness."""
    rows = []
    for conf, is_right in zip(confidences, correct):
        rows.append([conf, 1 - conf] if is_right else [1 - conf, conf])
    return np.array(rows)


def test_decide_routes_on_threshold():
    assert decide(0.91, 0.90) == AUTO
    assert decide(0.90, 0.90) == AUTO        # boundary is inclusive
    assert decide(0.89, 0.90) == ESCALATE


def test_higher_threshold_never_increases_coverage():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 400)
    correct = rng.random(400) < conf
    y_true = np.array(["a"] * 400)

    coverages = [p.coverage
                 for p in sweep(y_true, proba_from(conf, correct), CLASSES, 0.99)]
    assert coverages == sorted(coverages, reverse=True)


def test_best_operating_point_maximises_coverage_within_floor():
    # 10 confident-and-correct, 10 unconfident-and-wrong.
    conf = np.r_[np.full(10, 0.95), np.full(10, 0.60)]
    correct = np.r_[np.full(10, True), np.full(10, False)]
    y_true = np.array(["a"] * 20)
    point = best_operating_point(y_true, proba_from(conf, correct), CLASSES, 0.99)

    assert point is not None
    assert point.precision >= 0.99
    assert point.coverage == pytest.approx(0.5)    # exactly the confident half
    assert point.n_errors_covered == 0


def test_unreachable_floor_returns_none_rather_than_raising():
    # A model that is wrong half the time at every confidence level.
    conf = np.full(20, 0.80)
    correct = np.array([True, False] * 10)
    y_true = np.array(["a"] * 20)
    assert best_operating_point(y_true, proba_from(conf, correct), CLASSES, 0.99) is None


def test_flag_precision_collapses_when_the_event_is_rare():
    """99% classification precision is not 99% flag precision."""
    assert flag_precision(0.99, 0.01) == pytest.approx(0.5, abs=0.02)
    # Rarer events make flags worse, more common ones make them better.
    assert flag_precision(0.99, 0.002) < flag_precision(0.99, 0.05)


def test_flag_precision_is_one_when_the_classifier_is_perfect():
    assert flag_precision(1.0, 0.01) == pytest.approx(1.0)


def test_flag_precision_handles_degenerate_input():
    assert flag_precision(0.0, 0.0) == 0.0
