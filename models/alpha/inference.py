from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from config.schema import QuantusConfig
from models.alpha.calibration import apply_temperature, load_temperature
from models.alpha.dataset import load_scaler
from models.alpha.model import StockClassifier


@dataclass
class AlphaPrediction:
    predicted_class: int
    class_probabilities: List[float]
    confidence: float
    raw_logits: List[float]
    temperature: float


class AlphaInferenceService:
    def __init__(self, config: QuantusConfig):
        self.config = config
        self.scaler = load_scaler(config.paths.scaler_path)
        self.temperature = load_temperature(config.paths.temperature_path)
        self.model: Optional[StockClassifier] = None

    def load_model(self, input_dim: int) -> StockClassifier:
        model = StockClassifier(
            input_dim=input_dim,
            hidden1=self.config.alpha.hidden1,
            hidden2=self.config.alpha.hidden2,
            output_dim=self.config.alpha.output_dim,
            dropout=self.config.alpha.dropout,
        )
        state_dict = torch.load(self.config.paths.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        return model

    def prepare_input(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> torch.Tensor:
        latest = df[feature_columns].tail(self.config.features.lookback).copy()
        latest[feature_columns] = self.scaler.transform(latest[feature_columns])

        x = latest.to_numpy(dtype=np.float32, copy=True).flatten()[None, :]
        return torch.from_numpy(x).float()

    def predict(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> AlphaPrediction:
        x = self.prepare_input(df, feature_columns)

        model = self.model if self.model is not None else self.load_model(input_dim=x.shape[1])

        with torch.no_grad():
            logits = model(x)
            scaled_logits = apply_temperature(logits, self.temperature)
            probs = torch.softmax(scaled_logits, dim=1).squeeze(0)

        predicted_class = int(torch.argmax(probs).item())
        confidence = float(torch.max(probs).item())

        return AlphaPrediction(
            predicted_class=predicted_class,
            class_probabilities=probs.tolist(),
            confidence=confidence,
            raw_logits=logits.squeeze(0).tolist(),
            temperature=self.temperature,
        )
