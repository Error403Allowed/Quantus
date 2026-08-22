import torch
import torch.nn as nn
import torch.optim as optim


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = nn.Parameter(torch.tensor(1.0))
    optimiser = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimiser.zero_grad()
        loss = nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(temperature.item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature


def save_temperature(temperature: float, path: str) -> None:
    torch.save({"temperature": temperature}, path)


def load_temperature(path: str) -> float:
    obj = torch.load(path, map_location="cpu")
    return float(obj["temperature"])
