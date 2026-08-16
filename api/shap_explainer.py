"""
SHAP Explainability — Upgrade #6.

Provides pixel-level feature attribution using SHAP (SHapley Additive exPlanations).
Complements Grad-CAM++ with theoretically grounded importance scores.
"""

import base64
import io

import numpy as np

try:
    import matplotlib
    import shap
    import tensorflow as tf
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP DeepExplainer for brain tumor MRI classification.

    Uses a background dataset (small subset of training images) to compute
    Shapley values for each pixel, showing which regions contribute most
    to the prediction.
    """

    def __init__(self, model, background_images=None):
        if not SHAP_AVAILABLE:
            raise ImportError("shap package required. Install: pip install shap")

        self.model = model
        # Use a small background set (50-100 images) for efficiency
        if background_images is not None:
            self.background = background_images[:100]
        else:
            # Generate synthetic background (zeros) if no real data available
            self.background = np.zeros((50, 240, 240, 3), dtype=np.float32)

        self.explainer = shap.DeepExplainer(model, self.background)

    def explain(self, image: np.ndarray) -> str:
        """
        Generate SHAP explanation for a single image.

        Returns base64-encoded PNG of the SHAP visualization.
        """
        img_batch = np.expand_dims(image, 0).astype(np.float32)
        shap_values = self.explainer.shap_values(img_batch)

        # Get predicted class
        pred = self.model.predict(img_batch, verbose=0)
        pred_class = int(np.argmax(pred[0]))

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        orig = np.uint8((image - image.min()) / (image.max() - image.min() + 1e-7) * 255)
        axes[0].imshow(orig)
        axes[0].set_title("Original MRI", fontsize=12)
        axes[0].axis("off")

        # SHAP values for predicted class
        sv = shap_values[pred_class][0]
        # Aggregate across color channels
        sv_agg = sv.sum(axis=-1)

        # Positive contributions (supports prediction)
        axes[1].imshow(orig)
        pos_mask = np.maximum(sv_agg, 0)
        if pos_mask.max() > 0:
            pos_mask = pos_mask / pos_mask.max()
        axes[1].imshow(pos_mask, cmap="Reds", alpha=0.6)
        axes[1].set_title("Supporting Regions", fontsize=12)
        axes[1].axis("off")

        # Negative contributions (opposes prediction)
        axes[2].imshow(orig)
        neg_mask = np.maximum(-sv_agg, 0)
        if neg_mask.max() > 0:
            neg_mask = neg_mask / neg_mask.max()
        axes[2].imshow(neg_mask, cmap="Blues", alpha=0.6)
        axes[2].set_title("Opposing Regions", fontsize=12)
        axes[2].axis("off")

        plt.suptitle("SHAP Feature Attribution", fontsize=14, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def explain_all_classes(self, image: np.ndarray) -> str:
        """Generate SHAP explanation showing attribution for all 4 classes."""
        from api.model_service import CLASS_NAMES

        img_batch = np.expand_dims(image, 0).astype(np.float32)
        shap_values = self.explainer.shap_values(img_batch)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        orig = np.uint8((image - image.min()) / (image.max() - image.min() + 1e-7) * 255)
        axes[0].imshow(orig)
        axes[0].set_title("Original MRI", fontsize=11)
        axes[0].axis("off")

        for i, (idx, name) in enumerate(CLASS_NAMES.items()):
            sv = shap_values[idx][0].sum(axis=-1)
            axes[i + 1].imshow(orig)
            mask = np.where(sv > 0, sv, 0)
            if mask.max() > 0:
                mask = mask / mask.max()
            axes[i + 1].imshow(mask, cmap="hot", alpha=0.5)
            axes[i + 1].set_title(f"{name}", fontsize=11)
            axes[i + 1].axis("off")

        plt.suptitle("Per-Class SHAP Attribution", fontsize=13, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
