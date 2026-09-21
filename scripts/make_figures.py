"""Render the README figures from reports/metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "reports" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e8e7e3"
SERIES_1 = "#2a78d6"   # validated categorical slot 1
SERIES_2 = "#eb6834"   # validated categorical slot 2
MUTED = "#b8b7b0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "figure.dpi": 200,
})


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=13, fontweight="semibold",
                 loc="left", pad=14)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def coverage_tradeoff(m):
    sweep = m["threshold_sweep"]
    point = m["operating_points"]["precision_0.99"]
    x = [p["threshold"] for p in sweep]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, [p["coverage"] for p in sweep], lw=2, color=SERIES_1,
            label="Coverage (share auto-resolved)")
    ax.plot(x, [p["precision"] for p in sweep], lw=2, color=SERIES_2,
            label="Precision on covered volume")
    ax.axhline(0.99, lw=1.2, ls="--", color=MUTED, zorder=1)
    ax.text(0.505, 0.995, "99% precision floor", color=INK_SOFT, fontsize=9, va="bottom")

    if point:
        ax.plot([point["threshold"]], [point["coverage"]], "o", ms=8,
                color=SERIES_1, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(
            f"operating point\nthreshold {point['threshold']}\n"
            f"{point['coverage']:.1%} of volume at {point['precision']:.1%}",
            xy=(point["threshold"], point["coverage"]),
            xytext=(point["threshold"] + 0.05, point["coverage"] + 0.16),
            color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))

    _style(ax, "Raising the confidence threshold trades volume for precision",
           "Confidence threshold", "Fraction")
    ax.set_ylim(0, 1.05); ax.set_xlim(0.5, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    leg = ax.legend(frameon=False, loc="lower left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    fig.tight_layout(); fig.savefig(FIGURES / "coverage_tradeoff.png"); plt.close(fig)


def learning_curve(m):
    lc = m["learning_curve"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(lc["train_sizes"], lc["cv_accuracy"], lw=2, marker="o", ms=7,
            color=SERIES_1, mec=SURFACE, mew=2)
    last_x, last_y = lc["train_sizes"][-1], lc["cv_accuracy"][-1]
    ax.annotate(f"{last_y:.3f}", xy=(last_x, last_y), xytext=(-6, 12),
                textcoords="offset points", color=INK, fontsize=9, ha="right")
    _style(ax, "Accuracy flattens well before the data runs out",
           "Training documents", "5-fold CV accuracy")
    ax.set_ylim(0.6, 0.88)
    fig.tight_layout(); fig.savefig(FIGURES / "learning_curve.png"); plt.close(fig)


def baselines(m):
    rows = [("Majority class", m["baselines"]["majority_class"], MUTED),
            ("Hand-written regex", m["baselines"]["keyword_regex"], MUTED),
            ("TF-IDF + logistic regression", m["model"]["cv_accuracy_mean"], SERIES_1)]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    for i, (label, value, color) in enumerate(rows):
        ax.barh(i, value, height=0.55, color=color)
        ax.text(value + 0.012, i, f"{value:.3f}", va="center",
                color=INK, fontsize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis(); ax.set_xlim(0, 1.08); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style(ax, "", "Accuracy", "")
    # Title is anchored to the figure, not the axes: the category labels are wide
    # enough that an axes-anchored title overflows the canvas.
    fig.suptitle("The model earns 11 points over a regex a human could write",
                 color=INK, fontsize=13, fontweight="semibold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES / "baselines.png"); plt.close(fig)


def flag_precision(m):
    items = sorted((float(r), p) for r, p in m["flag_precision_by_base_rate"].items())
    xs = [r * 100 for r, _ in items]
    ys = [p for _, p in items]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(xs, ys, lw=2, marker="o", ms=7, color=SERIES_1, mec=SURFACE, mew=2)
    ax.axhline(0.99, lw=1.2, ls="--", color=MUTED, zorder=1)
    ax.text(0.21, 1.0, "classification precision (99%)", color=INK_SOFT,
            fontsize=9, va="bottom")
    idx = xs.index(1.0)
    ax.annotate(f"at a realistic 1% off-topic rate,\nonly {ys[idx]:.0%} of flags are real",
                xy=(xs[idx], ys[idx]), xytext=(1.5, 0.30), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))
    _style(ax, "Classification precision is not flag precision",
           "Genuine off-topic rate (% of posts)", "Precision of an off-topic flag")
    ax.set_xscale("log"); ax.set_ylim(0, 1.1)
    ax.set_xticks(xs, [f"{x:g}%" for x in xs])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout(); fig.savefig(FIGURES / "flag_precision.png"); plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    coverage_tradeoff(m); learning_curve(m); baselines(m); flag_precision(m)
    for f in sorted(FIGURES.glob("*.png")):
        print(f"  {f.relative_to(ROOT)}  ({f.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
