"""Tests for the evaluation metrics — pure maths, no model required."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    classification_report,
    confusion_matrix,
    format_report_markdown,
    report_to_dict,
    wilson_interval,
)

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


class TestWilsonInterval:
    def test_matches_published_value(self):
        # Known reference: 84.26% of 1311 → roughly 82.2%–86.1%.
        interval = wilson_interval(1105, 1311)
        assert interval.estimate == pytest.approx(0.8428, abs=1e-3)
        assert interval.low == pytest.approx(0.8213, abs=2e-3)
        assert interval.high == pytest.approx(0.8617, abs=2e-3)

    def test_interval_brackets_the_estimate(self):
        interval = wilson_interval(42, 100)
        assert interval.low < interval.estimate < interval.high

    def test_stays_within_unit_range_at_zero(self):
        interval = wilson_interval(0, 20)
        assert interval.low == 0.0
        assert 0 < interval.high < 1

    def test_stays_within_unit_range_at_one(self):
        # The normal approximation would produce a nonsensical upper bound above 1 here.
        interval = wilson_interval(20, 20)
        assert interval.high == 1.0
        assert 0 < interval.low < 1

    def test_small_samples_give_wide_intervals(self):
        narrow = wilson_interval(800, 1000)
        wide = wilson_interval(8, 10)
        assert wide.margin > narrow.margin

    def test_margin_shrinks_with_sample_size(self):
        margins = [wilson_interval(int(0.8 * n), n).margin for n in (50, 500, 5000)]
        assert margins[0] > margins[1] > margins[2]

    def test_higher_confidence_widens_interval(self):
        assert wilson_interval(80, 100, 0.99).margin > wilson_interval(80, 100, 0.90).margin

    def test_zero_samples_is_not_an_error(self):
        interval = wilson_interval(0, 0)
        assert interval.n == 0
        assert interval.as_percent() == "n/a"

    def test_rejects_successes_above_n(self):
        with pytest.raises(ValueError, match="between 0 and n"):
            wilson_interval(11, 10)

    def test_rejects_unsupported_confidence(self):
        with pytest.raises(ValueError, match="confidence must be"):
            wilson_interval(5, 10, 0.80)

    def test_as_percent_formatting(self):
        assert wilson_interval(50, 100).as_percent(1).startswith("50.0% [")


class TestConfusionMatrix:
    def test_perfect_predictions_are_diagonal(self):
        labels = [0, 1, 2, 3, 0, 1]
        matrix = confusion_matrix(labels, labels, 4)
        assert np.array_equal(matrix, np.diag([2, 2, 1, 1]))

    def test_counts_land_in_correct_cell(self):
        matrix = confusion_matrix([0, 0], [1, 1], 4)
        assert matrix[0, 1] == 2
        assert matrix.sum() == 2

    def test_rows_are_truth_columns_are_predictions(self):
        matrix = confusion_matrix([2], [0], 4)
        assert matrix[2, 0] == 1
        assert matrix[0, 2] == 0

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            confusion_matrix([0, 1], [0], 4)

    def test_rejects_out_of_range_labels(self):
        with pytest.raises(ValueError, match="outside"):
            confusion_matrix([4], [0], 4)


class TestClassificationReport:
    @staticmethod
    def _skewed():
        """20 gliomas of which 5 are called meningioma, plus 20 clean meningiomas."""
        y_true = [0] * 20 + [1] * 20
        y_pred = [0] * 15 + [1] * 5 + [1] * 20
        return y_true, y_pred

    def test_accuracy_carries_an_interval(self):
        report = classification_report(*self._skewed(), CLASSES)
        assert report["accuracy"].estimate == pytest.approx(35 / 40)
        assert report["accuracy"].low < report["accuracy"].high

    def test_glioma_recall_reflects_missed_cases(self):
        report = classification_report(*self._skewed(), CLASSES)
        glioma = report["per_class"][0]
        assert glioma.sensitivity.estimate == pytest.approx(0.75)
        assert glioma.false_negatives == 5

    def test_recall_and_precision_differ_for_skewed_errors(self):
        # Glioma is never over-called, so its precision is perfect despite poor recall.
        report = classification_report(*self._skewed(), CLASSES)
        glioma = report["per_class"][0]
        assert glioma.precision.estimate == pytest.approx(1.0)
        assert glioma.sensitivity.estimate < glioma.precision.estimate

    def test_records_where_missed_cases_went(self):
        report = classification_report(*self._skewed(), CLASSES)
        assert report["per_class"][0].missed_to == {"Meningioma": 5}

    def test_absent_classes_have_zero_support(self):
        report = classification_report(*self._skewed(), CLASSES)
        assert report["per_class"][2].support == 0

    def test_macro_average_ignores_absent_classes(self):
        # Only Glioma (0.75) and Meningioma (1.0) appear, so macro recall is 0.875.
        report = classification_report(*self._skewed(), CLASSES)
        assert report["macro_sensitivity"] == pytest.approx(0.875)

    def test_macro_average_exposes_weak_class(self):
        # Overall accuracy is dominated by the majority class; macro recall is not.
        y_true = [2] * 90 + [0] * 10
        y_pred = [2] * 90 + [2] * 10
        report = classification_report(y_true, y_pred, CLASSES)
        assert report["accuracy"].estimate == pytest.approx(0.90)
        assert report["macro_sensitivity"] == pytest.approx(0.50)

    def test_specificity_counts_true_negatives(self):
        report = classification_report(*self._skewed(), CLASSES)
        meningioma = report["per_class"][1]
        assert meningioma.false_positives == 5
        assert meningioma.specificity.estimate == pytest.approx(0.75)

    def test_perfect_classifier_scores_one(self):
        labels = [0, 1, 2, 3] * 10
        report = classification_report(labels, labels, CLASSES)
        assert report["accuracy"].estimate == 1.0
        assert report["macro_f1"] == pytest.approx(1.0)

    def test_confusion_matrix_totals_match_sample_count(self):
        report = classification_report(*self._skewed(), CLASSES)
        assert sum(sum(row) for row in report["confusion_matrix"]) == report["n_samples"]


class TestMarkdownFormatting:
    def test_includes_every_class(self):
        labels = [0, 1, 2, 3] * 5
        markdown = format_report_markdown(classification_report(labels, labels, CLASSES))
        for name in CLASSES:
            assert name in markdown

    def test_reports_missed_cases_section(self):
        y_true = [0] * 10 + [1] * 10
        y_pred = [1] * 10 + [1] * 10
        markdown = format_report_markdown(classification_report(y_true, y_pred, CLASSES))
        assert "Missed cases" in markdown
        assert "10 missed" in markdown

    def test_omits_missed_section_when_perfect(self):
        labels = [0, 1, 2, 3] * 5
        markdown = format_report_markdown(classification_report(labels, labels, CLASSES))
        assert "Missed cases" not in markdown

    def test_renders_confidence_level(self):
        labels = [0, 1, 2, 3] * 5
        markdown = format_report_markdown(classification_report(labels, labels, CLASSES))
        assert "95% Wilson CI" in markdown


class TestSerialization:
    def test_report_is_json_serializable(self):
        import json

        labels = [0, 1, 2, 3] * 5
        payload = report_to_dict(classification_report(labels, labels, CLASSES))
        assert json.loads(json.dumps(payload))["accuracy"]["estimate"] == 1.0

    def test_per_class_entries_survive_serialization(self):
        labels = [0, 1, 2, 3] * 5
        payload = report_to_dict(classification_report(labels, labels, CLASSES))
        assert payload["per_class"][0]["sensitivity"]["estimate"] == 1.0
