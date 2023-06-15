from typing import Any

import tensorflow_model_optimization as tfmot
from keras import Sequential

from .baseline_model import build_baseline_model


def build_pruned_model(pruning_params: dict[str, Any]) -> Sequential:
    baseline_model = build_baseline_model()
    return tfmot.sparsity.keras.prune_low_magnitude(baseline_model, **pruning_params)
