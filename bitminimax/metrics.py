from __future__ import annotations

import torch
from torch import Tensor, nn

from .games import exploitability


@torch.no_grad()
def binary_parameter_ratio(model: nn.Module) -> float:
    """Fraction of learned weight values that are used as binary forward weights."""
    total = binary = 0
    for module in model.modules():
        if hasattr(module, "weight") and isinstance(getattr(module, "weight"), Tensor):
            count = module.weight.numel()
            total += count
            binary += count if module.__class__.__name__ == "BinaryLinear" else 0
    return binary / max(total, 1)


@torch.no_grad()
def strategy_stability(model: nn.Module, game: Tensor, perturbation: float = 0.03) -> Tensor:
    row, col, _ = model(game)
    noise = torch.randn_like(game)
    noisy = game + perturbation * (noise - noise.transpose(-1, -2))
    noisy_row, noisy_col, _ = model(noisy)
    return (row - noisy_row).abs().sum(-1).add((col - noisy_col).abs().sum(-1)).mean()


@torch.no_grad()
def summarize(model: nn.Module, game: Tensor) -> dict[str, float]:
    row, col, _ = model(game)
    entropy = -(row * row.clamp_min(1e-8).log()).sum(-1).mean()
    return {"exploitability": exploitability(game, row, col).mean().item(), "entropy": entropy.item(), "stability_l1": strategy_stability(model, game).item(), "binary_parameter_ratio": binary_parameter_ratio(model)}
