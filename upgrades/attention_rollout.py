"""
Upgrade #7 — Attention Rollout for Vision Transformers.

When using the ViT model (upgrade #1), attention rollout provides
explainability by aggregating attention maps across all layers.
Shows which image patches the model attends to for its prediction.
"""

import numpy as np
import cv2
import tensorflow as tf


def get_attention_maps(vit_model, image: np.ndarray) -> list[np.ndarray]:
    """
    Extract attention maps from all transformer blocks.

    Returns a list of attention matrices, one per layer.
    Each has shape (num_heads, num_patches+1, num_patches+1).
    """
    # Build model that outputs attention weights from each transformer block
    attention_outputs = []
    img_batch = np.expand_dims(image, 0).astype(np.float32)

    # Forward pass collecting attention
    x = img_batch
    for layer in vit_model.layers:
        if hasattr(layer, "attn"):
            # MultiHeadAttention layers
            x_norm = layer.norm1(x)
            # Get attention scores
            attn_output, attn_weights = layer.attn(
                x_norm, x_norm, return_attention_scores=True
            )
            attention_outputs.append(attn_weights.numpy()[0])  # Remove batch dim
            x = x + attn_output
            x = x + layer.mlp(layer.norm2(x))
        elif hasattr(layer, "call"):
            x = layer(x)

    return attention_outputs


def attention_rollout(
    attention_maps: list[np.ndarray],
    head_fusion: str = "mean",
    discard_ratio: float = 0.1,
) -> np.ndarray:
    """
    Compute attention rollout across all layers.

    Multiplies attention matrices across layers to get the total
    attention flow from input patches to the [CLS] token.

    Args:
        attention_maps: List of attention matrices from each layer
        head_fusion: How to combine heads ("mean", "max", "min")
        discard_ratio: Fraction of lowest attention to zero out

    Returns:
        Attention mask of shape (num_patches,) — attention to [CLS]
    """
    result = None

    for attn in attention_maps:
        # Fuse heads
        if head_fusion == "mean":
            fused = attn.mean(axis=0)
        elif head_fusion == "max":
            fused = attn.max(axis=0)
        elif head_fusion == "min":
            fused = attn.min(axis=0)
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Discard low attention
        flat = fused.flatten()
        threshold = np.quantile(flat, discard_ratio)
        fused[fused < threshold] = 0

        # Add residual connection (identity matrix)
        I = np.eye(fused.shape[0])
        fused = (fused + I) / 2

        # Normalize rows
        fused = fused / fused.sum(axis=-1, keepdims=True)

        # Multiply through layers
        if result is None:
            result = fused
        else:
            result = result @ fused

    # Get attention from [CLS] token (first row) to all patches
    cls_attention = result[0, 1:]  # Exclude [CLS]-to-[CLS]
    return cls_attention


def visualize_attention_rollout(
    image: np.ndarray,
    attention_mask: np.ndarray,
    patch_size: int = 16,
    img_size: int = 240,
) -> np.ndarray:
    """
    Convert attention rollout mask to a visual heatmap overlaid on the image.

    Returns RGB overlay image.
    """
    num_patches_side = img_size // patch_size

    # Reshape to 2D grid
    mask = attention_mask.reshape(num_patches_side, num_patches_side)

    # Upscale to image size
    mask = cv2.resize(mask.astype(np.float32), (img_size, img_size))

    # Normalize
    if mask.max() > mask.min():
        mask = (mask - mask.min()) / (mask.max() - mask.min())

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    orig = np.uint8((image - image.min()) / (image.max() - image.min() + 1e-7) * 255)
    overlay = np.uint8(orig * 0.5 + heatmap * 0.5)

    return overlay


def explain_vit_prediction(vit_model, image: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Full attention rollout explanation pipeline.

    Returns (overlay_image, predicted_class_index).
    """
    # Get prediction
    pred = vit_model.predict(np.expand_dims(image, 0), verbose=0)
    pred_idx = int(np.argmax(pred[0]))

    # Get attention maps and compute rollout
    attn_maps = get_attention_maps(vit_model, image)
    mask = attention_rollout(attn_maps, head_fusion="mean", discard_ratio=0.1)
    overlay = visualize_attention_rollout(image, mask)

    return overlay, pred_idx
