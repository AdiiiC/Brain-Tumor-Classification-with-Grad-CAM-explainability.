"""
Grad-CAM++ implementation and visualization.
Corresponds to Notebook Cells 45-46.
"""
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

from config import CLASSES, CLASS_NAMES, TEST_DATA_DIR, IMG_SIZE


def gradcam_plus_plus(model, image):
    """
    Grad-CAM++: second-order gradient method giving sharper,
    more spatially accurate heatmaps than standard Grad-CAM.
    Returns an RGB heatmap resized to the input image dimensions.
    """
    last_conv = next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))
    grad_model = Model(model.inputs, [last_conv.output, model.output])

    img_batch = np.expand_dims(image, axis=0).astype('float32')

    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_out, preds = grad_model(img_batch)
                pred_idx = tf.argmax(preds[0])
                loss = preds[:, pred_idx]
            grads1 = tape1.gradient(loss, conv_out)
        grads2 = tape2.gradient(grads1, conv_out)
    grads3 = tape3.gradient(grads2, conv_out)

    # Alpha weights (Grad-CAM++ formula)
    global_sum = tf.reduce_sum(conv_out, axis=(1, 2), keepdims=True)
    alpha_denom = 2.0 * grads2 + global_sum * grads3 + 1e-7
    alpha = grads2 / alpha_denom
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads1), axis=(1, 2))[0]

    cam = tf.reduce_sum(weights * conv_out[0], axis=-1).numpy().astype(np.float32)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), int(pred_idx)


def visualize_gradcam(model, image, interpolant=0.5, class_dict=None):
    """
    Display a 4-panel figure:
      Original | Heatmap | Overlay | Confidence bar
    """
    assert 0 < interpolant < 1, "interpolant must be between 0 and 1"

    img_arr = np.array(image, dtype='float32')
    heatmap, pred_idx = gradcam_plus_plus(model, img_arr)

    orig_norm = np.uint8(
        (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-7) * 255
    )

    heatmap_rs = cv2.resize(heatmap, (orig_norm.shape[1], orig_norm.shape[0]))
    overlay = np.uint8(orig_norm * interpolant + heatmap_rs * (1 - interpolant))

    probs = model.predict(np.expand_dims(img_arr, 0), verbose=0)[0].astype(np.float32)
    labels = class_dict or {i: str(i) for i in range(len(probs))}

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].imshow(orig_norm)
    axes[0].set_title('Original MRI')
    axes[0].axis('off')

    axes[1].imshow(heatmap_rs)
    axes[1].set_title('Grad-CAM++ Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    bar_colors = ['tomato' if i == pred_idx else 'steelblue' for i in range(len(probs))]
    axes[3].barh([labels[i] for i in range(len(probs))], probs, color=bar_colors)
    axes[3].set_xlim(0, 1)
    axes[3].set_xlabel('Confidence')
    axes[3].set_title(f'Prediction: {labels[pred_idx]} ({probs[pred_idx]*100:.1f}%)')

    plt.suptitle('Grad-CAM++ Explainability', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return overlay


def run_gradcam_all_classes(model):
    """Run Grad-CAM++ on one sample from each class."""
    import os
    for cls_idx, cls_name in CLASS_NAMES.items():
        cls_dir = TEST_DATA_DIR / CLASSES[cls_idx]
        img_files = [f for f in os.listdir(str(cls_dir)) if not f.startswith('.')]
        if not img_files:
            continue
        img_path = cls_dir / img_files[0]
        test_img = cv2.imread(str(img_path))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = cv2.resize(test_img, IMG_SIZE)
        print(f"\n── {cls_name} ──")
        visualize_gradcam(model, test_img, interpolant=0.5, class_dict=CLASS_NAMES)
