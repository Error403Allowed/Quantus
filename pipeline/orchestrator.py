from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from config.schema import QuantusConfig
from models.alpha.inference import AlphaInferenceService


@dataclass
class OrchestratorDecision:
    action: str
    confidence: float
    model_outputs: Dict[str, Any]


class QuantusOrchestrator:
    def __init__(self, config: QuantusConfig):
        self.config = config
        self.alpha_service = AlphaInferenceService(config)

    def decide_from_alpha_only(
        self,
        df: pd.DataFrame,
        feature_columns,
    ) -> OrchestratorDecision:
        alpha_pred = self.alpha_service.predict(df, feature_columns)

        class_to_action = {
            0: "short",
            1: "flat",
            2: "long",
        }

        action = class_to_action[alpha_pred.predicted_class]

        return OrchestratorDecision(
            action=action,
            confidence=alpha_pred.confidence,
            model_outputs={
                "alpha": {
                    "predicted_class": alpha_pred.predicted_class,
                    "class_probabilities": alpha_pred.class_probabilities,
                    "confidence": alpha_pred.confidence,
                    "temperature": alpha_pred.temperature,
                }
            },
        )
