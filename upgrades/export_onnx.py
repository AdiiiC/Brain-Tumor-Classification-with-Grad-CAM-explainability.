"""
Upgrade #11 — ONNX Export.

Converts the Keras model to ONNX format for 2-5× faster CPU inference.
ONNX Runtime has optimized execution providers for CPU, GPU, and
specialized hardware (TensorRT, OpenVINO, DirectML).
"""

from pathlib import Path

import numpy as np


def export_to_onnx(
    keras_model_path: str = "model_best.keras",
    output_path: str = "brain_tumor_model.onnx",
    opset_version: int = 13,
    include_logits: bool = True,
) -> str:
    """
    Convert Keras model to ONNX format.

    With include_logits (the default) the graph emits both probabilities and pre-softmax
    logits. The serving layer needs logits for temperature calibration and for the
    free-energy OOD score — the latter is undefined from probabilities alone, since
    log-probabilities always sum-exp to 1.

    Requires: pip install tf2onnx onnx onnxruntime

    Returns the output file path.
    """
    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(keras_model_path)

    if include_logits:
        model = _with_logit_output(model)

    # Convert
    input_signature = [tf.TensorSpec(shape=(None, 240, 240, 3), dtype=tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset_version,
        output_path=output_path,
    )

    print(f"ONNX model saved: {output_path}")
    print(f"Size: {Path(output_path).stat().st_size / 1e6:.1f} MB")
    return output_path


def _with_logit_output(model):
    """Wrap the model so it returns (probabilities, logits)."""
    import tensorflow as tf

    final = model.layers[-1]
    logits = tf.keras.layers.Dense(final.units, activation=None, name="logits")(
        model.layers[-2].output
    )
    dual = tf.keras.Model(model.inputs, [model.output, logits])
    dual.get_layer("logits").set_weights(final.get_weights())
    return dual


def verify_onnx(onnx_path: str = "brain_tumor_model.onnx") -> dict:
    """
    Verify the ONNX model works and benchmark inference speed.
    """
    import time

    import onnxruntime as ort

    # Create session
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Test inference
    dummy_input = np.random.randn(1, 240, 240, 3).astype(np.float32)
    result = sess.run([output_name], {input_name: dummy_input})[0]

    assert result.shape == (1, 4), f"Unexpected output shape: {result.shape}"

    # Benchmark
    n_runs = 100
    start = time.time()
    for _ in range(n_runs):
        sess.run([output_name], {input_name: dummy_input})
    elapsed = time.time() - start

    avg_ms = (elapsed / n_runs) * 1000

    print("ONNX verification passed!")
    print(f"Output shape: {result.shape}")
    print(f"Average inference: {avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS)")

    return {
        "valid": True,
        "output_shape": result.shape,
        "avg_inference_ms": avg_ms,
        "fps": 1000 / avg_ms,
    }


class ONNXPredictor:
    """Fast ONNX Runtime predictor for production inference."""

    def __init__(self, model_path: str = "brain_tumor_model.onnx"):
        import onnxruntime as ort

        # Use available providers (GPU if available, else CPU)
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed image (240×240×3)."""
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        image = image.astype(np.float32)
        result = self.session.run([self.output_name], {self.input_name: image})[0]
        return result[0]

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Run batch inference."""
        images = images.astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: images})[0]


if __name__ == "__main__":
    export_to_onnx()
    verify_onnx()
