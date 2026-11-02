from dataclasses import dataclass
from typing import List

import pandas as pd
import torch

from config.schema import QuantusConfig
from models.alpha.calibration import apply_temperature, load_temperature
from models.alpha.dataset import load_metadata, load_scaler, make_inference_window
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

        meta = load_metadata(config.paths.metadata_path)
        self.feature_columns: List[str] = meta["feature_columns"]
        self.lookback: int = meta["lookback"]

        self.model = StockClassifier(
            input_dim=meta["input_dim"],
            hidden1=config.alpha.hidden1,
            hidden2=config.alpha.hidden2,
            output_dim=config.alpha.output_dim,
            dropout=config.alpha.dropout,
        )
        state_dict = torch.load(config.paths.model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, df: pd.DataFrame) -> AlphaPrediction:
        window = make_inference_window(df, self.feature_columns, self.lookback, self.scaler)
        x = torch.from_numpy(window).float()

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(apply_temperature(logits, self.temperature), dim=1).squeeze(0)

        predicted_class = int(torch.argmax(probs).item())
        return AlphaPrediction(
            predicted_class=predicted_class,
            class_probabilities=probs.tolist(),
            confidence=float(probs.max().item()),
            raw_logits=logits.squeeze(0).tolist(),
            temperature=self.temperature,
        )
    