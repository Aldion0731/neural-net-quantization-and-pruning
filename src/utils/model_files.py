from enum import Enum
from pathlib import Path


class ModelFiles(str, Enum):
    BASELINE_MODEL_WEIGHTS = "baseline_weights.h5"
    NON_QUANTIZED_H5 = "non_quantized.h5"
    NON_QUANTIZED_TFLITE = "non_quantized.tflite"
    POST_TRAINING_QUANTIZED_TFLITE = "post_training_quantized.tflite"
    QUANTIZED_AWARE_TRAINED_TFLITE = "quantization_aware_trained.tflite"
    PRUNED_MODEL_H5 = "pruned_model.h5"
    PRUNED_QUANTIZED_TFLITE = "pruned_quantized.tflite"


def calculate_file_path(file: ModelFiles, root_path: Path = Path("models_dir")) -> Path:
    return root_path / f"{file}"
