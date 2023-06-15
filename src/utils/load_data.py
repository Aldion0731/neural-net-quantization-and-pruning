from dataclasses import dataclass

import numpy as np
from keras.datasets import mnist


@dataclass
class MnistData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    def preprocess(self) -> None:
        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255


def load_data() -> MnistData:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return MnistData(X_train, X_test, y_train, y_test)
