from pydantic import BaseModel
import yaml
import os

class DataConfig(BaseModel):
    period: str
    interval: str
    min_rows: int

class ModelConfig(BaseModel):
    path: str
    scaler_path: str
    input_dim: int
    use_rag: bool

class LLMConfig(BaseModel):
    provider: str
    model: str
    enabled: bool

class ThresholdConfig(BaseModel):
    buy_label_return: float
    sell_label_return: float
    forward_days: int

class AppConfig(BaseModel):
    data: DataConfig
    news_provider: str
    sentiment_model: str
    model: ModelConfig
    llm: LLMConfig
    thresholds: ThresholdConfig
    finnhub_api_key: str 
    groq_api_key: str


def load_config(path="config/default.yaml") -> AppConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    local = "config/local.yaml"
    if os.path.exists(local):
        with open(local) as f:
            data.update(yaml.safe_load(f) or {})
    return AppConfig(**data)
