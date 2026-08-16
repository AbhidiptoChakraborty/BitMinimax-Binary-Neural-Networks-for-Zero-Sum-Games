from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .games import ZeroSumGameDistribution, expected_payoff, exploitability
from .games import transform_game
from .solvers import mirror_prox


@dataclass
class TrainConfig:
    actions: int = 7
    hidden_dim: int = 192
    depth: int = 3
    heads: int = 4
    batch_size: int = 256
    epochs: int = 400
    learning_rate: float = 3e-4
    entropy_weight: float = 2e-3
    consistency_weight: float = 0.15
    distillation_weight: float = 0.35
    robustness_weight: float = 0.10
    permutation_weight: float = 0.12
    solver_steps: int = 32
    perturbation_scale: float = 0.05
    grad_clip: float = 1.0
    device: str = "cpu"


class MinimaxTrainer:
    def __init__(self, model: nn.Module, games: ZeroSumGameDistribution, config: TrainConfig) -> None:
        self.model, self.games, self.config = model, games, config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    @staticmethod
    def _entropy(strategy: Tensor) -> Tensor:
        return -(strategy * strategy.clamp_min(1e-8).log()).sum(dim=-1)

    def loss(self, game: Tensor, difficulty: float = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
        row, col, _ = self.model(game)
        nashconv = exploitability(game, row, col)
        payoff = expected_payoff(game, row, col)

        # For anti-symmetric games, swapping players and transposing A should
        # swap policies. This makes the learned solver structurally coherent.
        swap_row, swap_col, _ = self.model(-game.transpose(-1, -2))
        symmetry_error = F.mse_loss(row, swap_col) + F.mse_loss(col, swap_row)
        entropy = self._entropy(row).mean() + self._entropy(col).mean()
        with torch.no_grad():
            target_row, target_col = mirror_prox(game, self.config.solver_steps)
        distillation = F.kl_div(row.clamp_min(1e-8).log(), target_row, reduction="batchmean") + F.kl_div(col.clamp_min(1e-8).log(), target_col, reduction="batchmean")
        perturbed = transform_game(game, noise_scale=self.config.perturbation_scale * difficulty)
        robust_row, robust_col, _ = self.model(perturbed)
        robustness = F.mse_loss(row, robust_row) + F.mse_loss(col, robust_col)
        permutation = torch.randperm(game.shape[-1], device=game.device)
        relabeled = transform_game(game, permutation=permutation)
        relabeled_row, relabeled_col, _ = self.model(relabeled)
        # New action i denotes old action permutation[i]; scatter restores old labels.
        restore_index = permutation.expand(game.shape[0], -1)
        aligned_row = torch.zeros_like(relabeled_row).scatter(1, restore_index, relabeled_row)
        aligned_col = torch.zeros_like(relabeled_col).scatter(1, restore_index, relabeled_col)
        permutation_error = F.mse_loss(row, aligned_row) + F.mse_loss(col, aligned_col)
        objective = nashconv.mean() + self.config.consistency_weight * symmetry_error + self.config.distillation_weight * distillation + self.config.robustness_weight * robustness + self.config.permutation_weight * permutation_error - self.config.entropy_weight * entropy
        metrics = {
            "loss": objective.detach(),
            "exploitability": nashconv.mean().detach(),
            "payoff_abs": payoff.abs().mean().detach(),
            "symmetry_error": symmetry_error.detach(),
            "entropy": entropy.detach(),
            "distillation": distillation.detach(),
            "robustness": robustness.detach(),
            "permutation_error": permutation_error.detach(),
        }
        return objective, metrics

    def train_epoch(self, epoch: int = 1) -> dict[str, float]:
        self.model.train()
        difficulty = min(1.0, 0.2 + 0.8 * epoch / max(self.config.epochs * 0.55, 1))
        game = self.games.sample(self.config.batch_size, self.config.device, difficulty)
        loss, metrics = self.loss(game, difficulty)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return {key: value.item() for key, value in metrics.items()}

    @torch.no_grad()
    def evaluate(self, batches: int = 16) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        for _ in range(batches):
            _, metrics = self.loss(self.games.sample(self.config.batch_size, self.config.device))
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value.item()
        return {key: value / batches for key, value in totals.items()}

    def checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "config": asdict(self.config), "metrics": metrics}, path)
