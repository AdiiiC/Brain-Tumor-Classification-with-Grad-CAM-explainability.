"""Evaluation utilities: confusion matrices, per-class clinical metrics, confidence intervals."""

from evaluation.metrics import (
    ClassMetrics,
    Interval,
    classification_report,
    confusion_matrix,
    format_report_markdown,
    wilson_interval,
)

__all__ = [
    "ClassMetrics",
    "Interval",
    "classification_report",
    "confusion_matrix",
    "format_report_markdown",
    "wilson_interval",
]
