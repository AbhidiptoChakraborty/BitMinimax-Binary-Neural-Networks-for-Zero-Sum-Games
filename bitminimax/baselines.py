from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .solvers import mirror_prox


class GameSolver(Protocol):
    def __call__(self, game: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]: ...


class FullPrecisionMinimaxPolicy(nn.Module):
    """Architecture-matched FP32 control used for binary ablation studies."""

    def __init__(self, actions: int, hidden_dim: int = 192, depth: int = 3, heads: int = 4, temperature: float = 0.7) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.temperature = temperature
        self.action_embedding = nn.Parameter(torch.randn(actions, hidden_dim) * 0.02)
        self.row_projection = nn.Linear(actions, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, dim_feedforward=hidden_dim * 2, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)
        self.row_head, self.col_head = nn.Linear(hidden_dim, actions), nn.Linear(hidden_dim, actions)

    def forward(self, game: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        tokens = self.row_projection(game) + self.action_embedding.unsqueeze(0)
        features = self.norm(self.encoder(tokens)).mean(dim=1)
        row_logits, col_logits = self.row_head(features), self.col_head(features)
        return F.softmax(row_logits / self.temperature, -1), F.softmax(col_logits / self.temperature, -1), {"row_logits": row_logits, "col_logits": col_logits}


class MirrorProxOracle:
    """Non-learned reference solver for lower-bound and calibration comparisons."""

    def __init__(self, steps: int = 256, step_size: float = 0.35) -> None:
        self.steps, self.step_size = steps, step_size

    @torch.no_grad()
    def __call__(self, game: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        row, col = mirror_prox(game, self.steps, self.step_size)
        return row, col, {"solver_steps": torch.tensor(self.steps, device=game.device)}


class UniformPolicy:
    """Sanity-check policy that exposes whether a learned model beats chance."""

    @torch.no_grad()
    def __call__(self, game: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        strategy = torch.full_like(game[:, 0, :], 1 / game.shape[-1])
        return strategy, strategy, {}
