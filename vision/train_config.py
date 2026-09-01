from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class TrainSettings(BaseModel):
    faces_root: Path
    models_dir: Path
    batch_size: int
    num_epochs: int
    learning_rate: float
    val_split: float
    num_workers: int
    embedding_dim: int


TRAIN_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "config" / "train.yaml"


def load_train_settings() -> TrainSettings:
    raw = yaml.safe_load(TRAIN_CONFIG_PATH.read_text())
    raw["faces_root"] = Path(raw["faces_root"])
    raw["models_dir"] = Path(raw["models_dir"])
    return TrainSettings(**raw)
