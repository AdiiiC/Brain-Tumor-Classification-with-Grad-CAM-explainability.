"""
Evaluate the classifier and emit per-class metrics with confidence intervals.

Headline accuracy is not enough for a medical model: this reports sensitivity and
specificity per class, plus where missed cases actually went.

Usage:
    python -m scripts.evaluate --data-dir datasets/brain-tumor-mri-dataset/Testing
    python -m scripts.evaluate --data-dir <dir> --json results.json --markdown results.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from api.model_service import CLASS_NAMES, ModelService
from evaluation.metrics import classification_report, format_report_markdown, report_to_dict

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Dataset folders are named inconsistently across sources (glioma_tumor, glioma, g_tumor...).
_FOLDER_KEYWORDS = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "no_tumor": 2,
    "healthy": 2,
    "normal": 2,
    "pituitary": 3,
}


def label_for_folder(name: str) -> int | None:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    if normalized.replace("_", "") in ("notumor", "nontumor"):
        return 2
    for keyword, label in _FOLDER_KEYWORDS.items():
        if keyword in normalized:
            return label
    return None


def collect_images(data_dir: Path, limit_per_class: int | None) -> list[tuple[Path, int]]:
    """Find labelled images in class-named subdirectories."""
    items: list[tuple[Path, int]] = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = label_for_folder(folder.name)
        if label is None:
            print(f"  skipping unrecognized folder: {folder.name}", file=sys.stderr)
            continue
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if limit_per_class:
            files = files[:limit_per_class]
        items.extend((f, label) for f in files)
        print(f"  {folder.name} → {CLASS_NAMES[label]}: {len(files)} images", file=sys.stderr)
    return items


def evaluate(model: ModelService, items: list[tuple[Path, int]], calibrated: bool):
    y_true: list[int] = []
    y_pred: list[int] = []
    failures = 0

    for i, (path, label) in enumerate(items, 1):
        try:
            img = model.preprocess_image(path.read_bytes())
            probs = model.predict_calibrated(img) if calibrated else model.predict(img)
        except Exception as exc:  # a corrupt file should not abort a long run
            failures += 1
            print(f"  failed on {path.name}: {exc}", file=sys.stderr)
            continue

        y_true.append(label)
        y_pred.append(int(probs.argmax()))

        if i % 100 == 0 or i == len(items):
            print(f"  {i}/{len(items)} images", end="\r", file=sys.stderr)

    print(file=sys.stderr)
    return y_true, y_pred, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory of class subfolders")
    parser.add_argument("--model-path", default="model_best.keras")
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.95, choices=[0.90, 0.95, 0.99])
    parser.add_argument("--raw", action="store_true", help="Skip temperature calibration")
    parser.add_argument("--json", type=Path, help="Write the full report as JSON")
    parser.add_argument("--markdown", type=Path, help="Write a Markdown summary")
    parser.add_argument("--title", default="Evaluation")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"error: {args.data_dir} is not a directory", file=sys.stderr)
        return 1

    print(f"Loading model from {args.model_path}...", file=sys.stderr)
    model = ModelService(model_path=args.model_path)
    if not model.is_loaded:
        print("error: model could not be loaded (is TensorFlow installed?)", file=sys.stderr)
        return 1

    from api import config

    model.set_calibration_temperature(config.calibration_temp())

    print(f"Scanning {args.data_dir}...", file=sys.stderr)
    items = collect_images(args.data_dir, args.limit_per_class)
    if not items:
        print("error: no labelled images found", file=sys.stderr)
        return 1

    y_true, y_pred, failures = evaluate(model, items, calibrated=not args.raw)
    if not y_true:
        print("error: every image failed to process", file=sys.stderr)
        return 1

    report = classification_report(y_true, y_pred, list(CLASS_NAMES.values()), args.confidence)
    markdown = format_report_markdown(report, args.title)
    print(markdown)

    if failures:
        print(f"\n{failures} image(s) could not be processed.", file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps(report_to_dict(report), indent=2))
        print(f"\nJSON written to {args.json}", file=sys.stderr)
    if args.markdown:
        args.markdown.write_text(markdown)
        print(f"Markdown written to {args.markdown}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
