from dataclasses import dataclass

from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape


@dataclass
class CompileParams:
    optimizer: str
    loss: str
    metrics: list[str]


def build_baseline_model() -> Sequential:
    model = Sequential(
        [
            Input(shape=(28, 28)),
            Reshape(target_shape=(28, 28, 1)),
            Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )

    return model
