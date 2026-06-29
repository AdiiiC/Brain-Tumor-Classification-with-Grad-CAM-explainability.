"""
Gradio interactive web app for MRI classification.
Corresponds to Notebook Cell 49.
"""
import os

import cv2
import numpy as np
import PIL
import gradio as gr

from config import CLASSES, CLASS_NAMES, TEST_DATA_DIR, IMG_SIZE
from gradcam import gradcam_plus_plus
from uncertainty import predict_with_uncertainty


def create_gradio_app(model):
    """Create and return a Gradio interface for the model."""

    def predict_mri(pil_image):
        img = np.array(pil_image.convert('RGB').resize(IMG_SIZE))

        # Standard prediction
        probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx]) * 100

        # Monte Carlo uncertainty
        _, uncertainty = predict_with_uncertainty(model, img, n_iterations=30)
        unc_val = float(uncertainty[pred_idx]) * 100

        # Grad-CAM++ heatmap
        heatmap, _ = gradcam_plus_plus(model, img)
        heatmap_rs = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        overlay = np.uint8(img * 0.5 + heatmap_rs * 0.5)

        label = (
            f"Prediction: {CLASS_NAMES[pred_idx]}\n"
            f"Confidence: {confidence:.1f}%  ±{unc_val:.1f}%\n"
            + (" ⚠ Low confidence — consider specialist review" if unc_val > 5 else " ✓ High confidence")
        )

        return PIL.Image.fromarray(overlay), label

    # Get example images
    examples = []
    first_cls_dir = str(TEST_DATA_DIR / CLASSES[0])
    if os.path.exists(first_cls_dir):
        examples = [
            [str(TEST_DATA_DIR / CLASSES[0] / f)]
            for f in os.listdir(first_cls_dir)[:2]
            if not f.startswith('.')
        ]

    demo = gr.Interface(
        fn=predict_mri,
        inputs=gr.Image(type="pil", label="Upload MRI Scan"),
        outputs=[gr.Image(label="Grad-CAM++ Overlay"), gr.Textbox(label="Result")],
        title="Brain Tumor MRI Classifier",
        description="Upload an MRI scan → get tumor class, confidence, uncertainty, and Grad-CAM++ heatmap.",
        examples=examples
    )
    return demo
