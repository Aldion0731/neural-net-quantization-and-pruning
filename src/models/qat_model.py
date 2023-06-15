from pathlib import Path

import tensorflow_model_optimization as tfmot
from keras import Sequential

from .baseline_model import build_baseline_model


def build_qat_model(baseline_model_weights_path: Path) -> Sequential:
    model_to_quantize = build_baseline_model()
    model_to_quantize.load_weights(baseline_model_weights_path)
    return tfmot.quantization.keras.quantize_model(model_to_quantize)
