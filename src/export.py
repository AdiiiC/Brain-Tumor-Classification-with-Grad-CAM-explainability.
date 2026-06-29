"""
TFLite export with INT8 quantization.
Corresponds to Notebook Cell 50.
"""
import json

import numpy as np
import tensorflow as tf

from config import BASE_DIR


def export_tflite(model, output_path=None):
    """
    Export model to TFLite with INT8 post-training quantization.
    Handles mixed_float16 policy by converting to float32 first.
    """
    if output_path is None:
        output_path = BASE_DIR / "brain_tumor_model.tflite"

    print("Converting to TFLite with INT8 quantization ...")

    # Rebuild as float32 model (mixed_float16 stores weights as float32 anyway)
    _cfg_str = json.dumps(model.get_config()).replace('"mixed_float16"', '"float32"')
    model_fp32 = model.__class__.from_config(json.loads(_cfg_str))
    model_fp32.set_weights(model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(model_fp32)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output_path.write_bytes(tflite_model)
    print(f"TFLite model saved → {output_path}  "
          f"({output_path.stat().st_size / 1e6:.1f} MB)")

    return output_path


def verify_tflite(tflite_path):
    """Verify TFLite model runs correctly."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # Run one inference to confirm it works
    sample_img = np.zeros((1, 240, 240, 3), dtype=np.float32)
    interpreter.set_tensor(inp['index'], sample_img)
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(out['index'])
    print(f"TFLite test inference output shape: {tflite_pred.shape}  ✓")
    return tflite_pred
