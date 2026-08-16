"""
Fit the OOD detector on in-distribution training images.

Without fitted statistics the API falls back to energy scoring, which works but
is less discriminative. Run this once against the training set:

    python -m scripts.fit_ood --data-dir datasets/brain-tumor-mri-dataset/Training

The resulting ood_stats.npz is picked up automatically at startup
(override the location with OOD_STATS_PATH).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.model_service import CLASS_NAMES, ModelService  # noqa: E402
from api.ood import OODDetector  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(data_dir: Path) -> list[tuple[Path, int]]:
    """Walk class-named subdirectories and pair each image with its label index."""
    name_to_index = {name.lower().replace(" ", "").replace("_", ""): idx
                     for idx, name in CLASS_NAMES.items()}

    pairs: list[tuple[Path, int]] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        key = class_dir.name.lower().replace(" ", "").replace("_", "")
        label = name_to_index.get(key)
        if label is None:
            print(f"  skipping unrecognised class directory: {class_dir.name}")
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                pairs.append((image_path, label))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit OOD statistics from training images.")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory with one subdirectory per class")
    parser.add_argument("--model-path", default="model_best.keras")
    parser.add_argument("--output", default="ood_stats.npz")
    parser.add_argument("--max-per-class", type=int, default=400,
                        help="Cap images per class to keep fitting quick")
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="In-distribution percentile used as the rejection threshold")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"error: {args.data_dir} is not a directory")
        return 1

    service = ModelService(model_path=args.model_path)
    if service.model_type != "keras":
        print(f"error: could not load a Keras model from {args.model_path}")
        return 1

    pairs = collect_images(args.data_dir)
    if not pairs:
        print("error: no images found")
        return 1

    rng = np.random.default_rng(42)
    by_class: dict[int, list[Path]] = {}
    for path, label in pairs:
        by_class.setdefault(label, []).append(path)

    selected: list[tuple[Path, int]] = []
    for label, paths in by_class.items():
        if len(paths) > args.max_per_class:
            idx = rng.choice(len(paths), args.max_per_class, replace=False)
            paths = [paths[i] for i in idx]
        selected.extend((p, label) for p in paths)

    print(f"Extracting features from {len(selected)} images...")
    features, labels = [], []
    for count, (path, label) in enumerate(selected, start=1):
        try:
            img = service.preprocess_image(path.read_bytes())
            feature = service.extract_features(img)
        except Exception as exc:  # noqa: BLE001 - skip unreadable files
            print(f"  skipped {path.name}: {exc}")
            continue
        if feature is None:
            continue
        features.append(feature)
        labels.append(label)
        if count % 100 == 0:
            print(f"  {count}/{len(selected)}")

    if len(features) < 20:
        print("error: too few usable images to estimate a covariance")
        return 1

    features_arr = np.vstack(features)
    labels_arr = np.array(labels)

    detector = OODDetector(stats_path="__unfitted__.npz")
    detector.fit(features_arr, labels_arr)

    # Calibrate the threshold so the chosen percentile of training data is accepted.
    distances = np.array([detector.mahalanobis(f) for f in features_arr])
    detector.mahalanobis_threshold = float(np.percentile(distances, args.percentile))
    detector.save(args.output)

    print(f"\nFitted on {len(features_arr)} images across {len(set(labels))} classes.")
    print(f"In-distribution distance: median {np.median(distances):.2f}, "
          f"p{args.percentile:.0f} {detector.mahalanobis_threshold:.2f}")
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
