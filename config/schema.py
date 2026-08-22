from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class DataConfig:
    tickers: List[str]
    period: str
    interval: str


@dataclass
class FeaturesConfig:
    lookback: int
    horizon: int
    drop_columns: List[str]


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float


@dataclass
class AlphaConfig:
    hidden1: int
    hidden2: int
    dropout: float
    output_dim: int


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int


@dataclass
class PathsConfig:
    alpha_dir: str
    model_path: str
    scaler_path: str
    temperature_path: str


@dataclass
class QuantusConfig:
    data: DataConfig
    features: FeaturesConfig
    split: SplitConfig
    alpha: AlphaConfig
    training: TrainingConfig
    paths: PathsConfig


def load_config(path: str = "config/default.yaml") -> QuantusConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return QuantusConfig(
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        split=SplitConfig(**raw["split"]),
        alpha=AlphaConfig(**raw["alpha"]),
        training=TrainingConfig(**raw["training"]),
        paths=PathsConfig(**raw["paths"]),
    )
