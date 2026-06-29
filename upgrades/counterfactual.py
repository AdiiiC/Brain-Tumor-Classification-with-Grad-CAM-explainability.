"""
Upgrade #8 — Counterfactual Explanations.

Shows what would need to change in the MRI for a different classification.
"This scan is classified as Glioma — if this region were absent, it would be No Tumor."

Uses occlusion-based counterfactual generation:
- Systematically mask regions of the image
- Find the minimal mask that changes the prediction
- Highlight these critical regions for the doctor
"""

import numpy as np
import cv2
import tensorflow as tf
from itertools import product


def generate_counterfactual(
    model,
    image: np.ndarray,
    target_class: int | None = None,
    patch_size: int = 20,
    stride: int = 10,
    max_patches: int = 5,
) -> dict:
    """
    Find minimal set of patches whose removal changes the prediction.

    Args:
        model: Trained Keras model
        image: Input MRI (240×240×3)
        target_class: Which class to flip to (None = any different class)
        patch_size: Size of occlusion patches
        stride: Step between patches
        max_patches: Maximum patches to remove

    Returns:
        Dict with counterfactual explanation details
    """
    img_batch = np.expand_dims(image, 0).astype(np.float32)
    original_pred = model.predict(img_batch, verbose=0)[0]
    original_class = int(np.argmax(original_pred))

    h, w = image.shape[:2]
    importance_map = np.zeros((h, w), dtype=np.float32)

    # Phase 1: Compute importance of each patch
    positions = list(product(
        range(0, h - patch_size + 1, stride),
        range(0, w - patch_size + 1, stride)
    ))

    patch_scores = []
    for y, x in positions:
        masked = image.copy()
        masked[y:y + patch_size, x:x + patch_size] = 0  # Black out patch

        pred = model.predict(np.expand_dims(masked, 0), verbose=0)[0]
        # How much does masking this reduce the original class confidence?
        drop = original_pred[original_class] - pred[original_class]
        patch_scores.append((drop, y, x))

        # Accumulate importance
        importance_map[y:y + patch_size, x:x + patch_size] += drop

    # Sort by importance (highest drop first)
    patch_scores.sort(reverse=True)

    # Phase 2: Find minimal set that flips the prediction
    counterfactual_img = image.copy()
    removed_patches = []

    for i, (drop, y, x) in enumerate(patch_scores[:max_patches * 3]):
        counterfactual_img[y:y + patch_size, x:x + patch_size] = 0
        removed_patches.append((y, x))

        pred = model.predict(np.expand_dims(counterfactual_img, 0), verbose=0)[0]
        new_class = int(np.argmax(pred))

        if target_class is not None and new_class == target_class:
            break
        elif target_class is None and new_class != original_class:
            break

        if len(removed_patches) >= max_patches:
            break

    # Generate visualization
    final_pred = model.predict(np.expand_dims(counterfactual_img, 0), verbose=0)[0]
    final_class = int(np.argmax(final_pred))

    # Normalize importance map for visualization
    if importance_map.max() > 0:
        importance_map = importance_map / importance_map.max()

    return {
        "original_class": original_class,
        "original_confidence": float(original_pred[original_class]),
        "counterfactual_class": final_class,
        "counterfactual_confidence": float(final_pred[final_class]),
        "flipped": final_class != original_class,
        "patches_removed": len(removed_patches),
        "critical_regions": removed_patches,
        "importance_map": importance_map,
        "counterfactual_image": counterfactual_img,
    }


def visualize_counterfactual(
    image: np.ndarray,
    result: dict,
    class_names: dict,
) -> np.ndarray:
    """
    Create a side-by-side visualization:
    Original | Critical Regions | Counterfactual
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    orig = np.uint8((image - image.min()) / (image.max() - image.min() + 1e-7) * 255)

    # Original
    axes[0].imshow(orig)
    axes[0].set_title(
        f"Original: {class_names[result['original_class']]}\n"
        f"({result['original_confidence']*100:.1f}%)",
        fontsize=12
    )
    axes[0].axis("off")

    # Importance heatmap
    axes[1].imshow(orig)
    heatmap = cv2.applyColorMap(np.uint8(255 * result["importance_map"]), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    axes[1].imshow(heatmap, alpha=0.6)
    axes[1].set_title(f"Critical Regions\n({result['patches_removed']} patches)", fontsize=12)
    axes[1].axis("off")

    # Counterfactual
    cf = np.uint8((result["counterfactual_image"] - result["counterfactual_image"].min()) /
                  (result["counterfactual_image"].max() - result["counterfactual_image"].min() + 1e-7) * 255)
    axes[2].imshow(cf)
    status = "FLIPPED" if result["flipped"] else "NOT FLIPPED"
    color = "green" if result["flipped"] else "red"
    axes[2].set_title(
        f"Counterfactual: {class_names[result['counterfactual_class']]}\n"
        f"({result['counterfactual_confidence']*100:.1f}%) — {status}",
        fontsize=12, color=color
    )
    axes[2].axis("off")

    plt.suptitle("Counterfactual Explanation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    # Decode back to numpy for return
    arr = np.frombuffer(buf.read(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
