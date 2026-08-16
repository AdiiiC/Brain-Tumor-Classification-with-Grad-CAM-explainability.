"""
Classification metrics with confidence intervals.

Headline accuracy hides the failure mode that matters clinically: a missed glioma is
not equivalent to a false meningioma. This module reports per-class sensitivity and
specificity with Wilson score intervals so that small-sample results are not read as
more precise than they are.

Pure NumPy — importable without TensorFlow.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

# Two-sided normal quantiles for the confidence levels we support.
_Z_SCORES = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}

DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval, all on a 0-1 scale."""

    estimate: float
    low: float
    high: float
    n: int

    @property
    def margin(self) -> float:
        """Half-width of the interval, i.e. the '±' figure."""
        return (self.high - self.low) / 2

    def as_percent(self, decimals: int = 2) -> str:
        if self.n == 0:
            return "n/a"
        return f"{self.estimate * 100:.{decimals}f}% [{self.low * 100:.{decimals}f}–{self.high * 100:.{decimals}f}]"

    def to_dict(self) -> dict:
        return asdict(self)


def wilson_interval(
    successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE
) -> Interval:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and remains
    sensible for small n and for proportions near 0 or 1 — exactly the regime of
    per-class recall on a few hundred images.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0 <= successes <= n:
        raise ValueError(f"successes ({successes}) must be between 0 and n ({n})")
    if confidence not in _Z_SCORES:
        raise ValueError(f"confidence must be one of {sorted(_Z_SCORES)}")

    if n == 0:
        return Interval(estimate=0.0, low=0.0, high=0.0, n=0)

    z = _Z_SCORES[confidence]
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

    return Interval(
        estimate=p,
        low=max(0.0, center - half_width),
        high=min(1.0, center + half_width),
        n=n,
    )


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size and (y_true.min() < 0 or y_true.max() >= n_classes):
        raise ValueError("y_true contains labels outside [0, n_classes)")
    if y_pred.size and (y_pred.min() < 0 or y_pred.max() >= n_classes):
        raise ValueError("y_pred contains labels outside [0, n_classes)")

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


@dataclass
class ClassMetrics:
    """One-vs-rest metrics for a single class."""

    name: str
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    sensitivity: Interval  # recall — of the actual cases, how many were caught
    specificity: Interval  # of the non-cases, how many were correctly cleared
    precision: Interval  # PPV — of the positive calls, how many were right
    npv: Interval
    f1: float
    missed_to: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "support": self.support,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "sensitivity": self.sensitivity.to_dict(),
            "specificity": self.specificity.to_dict(),
            "precision": self.precision.to_dict(),
            "npv": self.npv.to_dict(),
            "f1": self.f1,
            "missed_to": self.missed_to,
        }


def _class_metrics(
    matrix: np.ndarray, index: int, class_names: list[str], confidence: float
) -> ClassMetrics:
    tp = int(matrix[index, index])
    fn = int(matrix[index].sum() - tp)
    fp = int(matrix[:, index].sum() - tp)
    tn = int(matrix.sum() - tp - fn - fp)

    precision = wilson_interval(tp, tp + fp, confidence)
    sensitivity = wilson_interval(tp, tp + fn, confidence)
    f1_denominator = precision.estimate + sensitivity.estimate
    f1 = 0.0 if f1_denominator == 0 else 2 * precision.estimate * sensitivity.estimate / f1_denominator

    missed_to = {
        class_names[j]: int(matrix[index, j])
        for j in range(len(class_names))
        if j != index and matrix[index, j] > 0
    }

    return ClassMetrics(
        name=class_names[index],
        support=tp + fn,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        sensitivity=sensitivity,
        specificity=wilson_interval(tn, tn + fp, confidence),
        precision=precision,
        npv=wilson_interval(tn, tn + fn, confidence),
        f1=f1,
        missed_to=missed_to,
    )


def classification_report(
    y_true,
    y_pred,
    class_names: list[str],
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """
    Full report: overall accuracy with a CI, per-class metrics, and the confusion matrix.

    Macro averages weight every class equally, so a rare-but-critical class cannot be
    hidden by good performance on a common one.
    """
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    total = int(matrix.sum())
    correct = int(np.trace(matrix))

    per_class = [_class_metrics(matrix, i, class_names, confidence) for i in range(len(class_names))]
    evaluated = [m for m in per_class if m.support > 0]

    def macro(attr: str) -> float:
        if not evaluated:
            return 0.0
        return float(np.mean([getattr(m, attr).estimate for m in evaluated]))

    return {
        "n_samples": total,
        "confidence_level": confidence,
        "accuracy": wilson_interval(correct, total, confidence),
        "macro_sensitivity": macro("sensitivity"),
        "macro_specificity": macro("specificity"),
        "macro_precision": macro("precision"),
        "macro_f1": float(np.mean([m.f1 for m in evaluated])) if evaluated else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
    }


def format_report_markdown(report: dict, title: str = "Evaluation") -> str:
    """Render a report as Markdown tables, ready to paste into the README."""
    accuracy: Interval = report["accuracy"]
    confidence_pct = int(report["confidence_level"] * 100)
    lines = [
        f"### {title}",
        "",
        f"**{report['n_samples']:,} images** · accuracy **{accuracy.as_percent()}** "
        f"({confidence_pct}% Wilson CI)",
        "",
        "| Class | Support | Sensitivity (recall) | Specificity | Precision (PPV) | F1 |",
        "|-------|---------|----------------------|-------------|-----------------|-----|",
    ]

    for metrics in report["per_class"]:
        lines.append(
            f"| {metrics.name} | {metrics.support:,} | {metrics.sensitivity.as_percent(1)} | "
            f"{metrics.specificity.as_percent(1)} | {metrics.precision.as_percent(1)} | "
            f"{metrics.f1:.3f} |"
        )

    lines += [
        f"| **Macro avg** | {report['n_samples']:,} | {report['macro_sensitivity'] * 100:.1f}% | "
        f"{report['macro_specificity'] * 100:.1f}% | {report['macro_precision'] * 100:.1f}% | "
        f"{report['macro_f1']:.3f} |",
        "",
        "Intervals are Wilson score intervals; ranges are shown in percentage points.",
        "",
        "#### Confusion matrix (rows = truth, columns = prediction)",
        "",
        "| | " + " | ".join(report["class_names"]) + " |",
        "|" + "---|" * (len(report["class_names"]) + 1),
    ]

    for name, row in zip(report["class_names"], report["confusion_matrix"], strict=True):
        lines.append(f"| **{name}** | " + " | ".join(str(v) for v in row) + " |")

    misses = [
        f"- **{m.name}**: {m.false_negatives} missed "
        f"({', '.join(f'{count} called {label}' for label, count in m.missed_to.items())})"
        for m in report["per_class"]
        if m.false_negatives > 0
    ]
    if misses:
        lines += ["", "#### Missed cases", ""] + misses

    return "\n".join(lines)


def report_to_dict(report: dict) -> dict:
    """JSON-serializable form of a report."""
    return {
        **report,
        "accuracy": report["accuracy"].to_dict(),
        "per_class": [m.to_dict() for m in report["per_class"]],
    }
