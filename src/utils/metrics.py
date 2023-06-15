import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Metrics:
    model_file_name: str
    model_size: float
    model_accuracy: float


class MetricsStore:
    def __init__(self) -> None:
        self.__store = []

    def update(self, metrics: Metrics) -> None:
        self.__store.append(metrics)

    def display(self) -> pd.DataFrame:
        table = pd.DataFrame(self.__store)
        return table.drop_duplicates()


def get_model_metrics(model_path: Path, model_accuracy: float) -> Metrics:
    return Metrics(model_path.name, model_path.stat().st_size, model_accuracy)


def compute_zipped_file_sizes(*args: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)

        for file in args:
            shutil.copy(file, tempdir_path)
            zip_file(tempdir_path, tempdir_path / file.name)

        zip_files = [
            tempdir_path / file
            for file in os.listdir(tempdir_path)
            if file.endswith(".zip")
        ]
        return compare_model_sizes(*zip_files)


def compare_model_sizes(*args: Path) -> pd.DataFrame:
    model_sizes = [path.stat().st_size for path in args]
    model_names = [path.name for path in args]
    return pd.DataFrame({"model": model_names, "size": model_sizes})


def zip_file(zip_root: Path, file: Path) -> None:
    zipped_path = zip_root / f"{file.stem}_{file.suffix.lstrip('.')}.zip"
    with zipfile.ZipFile(zipped_path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(file, file.name)
