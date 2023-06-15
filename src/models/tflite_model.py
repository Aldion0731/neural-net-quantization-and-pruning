from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import Model, Sequential


def convert_and_save_model_as_tflite(
    model: Model | Sequential, tflite_path: Path, quantize: bool = False
) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


class TFLiteModel:
    def __init__(self, tflite_path: Path) -> None:
        self.interpreter = tf.lite.Interpreter(str(tflite_path))
        self.interpreter.allocate_tensors()

    def predict(self, image: np.ndarray) -> int:
        input = self.interpreter.get_input_details()[0]
        image = np.expand_dims(image, axis=0).astype(np.float32)
        self.interpreter.set_tensor(input["index"], image)

        self.interpreter.invoke()

        output = self.interpreter.get_output_details()[0]
        predictions = self.interpreter.get_tensor(output["index"])
        return np.argmax(predictions, axis=1)[0]

    def evaluate(self, image_batch: np.ndarray, labels: np.ndarray) -> float:
        batch_predictions = []

        for image in image_batch:
            pred = self.predict(image)
            batch_predictions.append(pred)

        predictions = np.array(batch_predictions)
        return (predictions == labels).mean()
