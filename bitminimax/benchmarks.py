from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .games import ZeroSumGameDistribution
from .metrics import summarize


@dataclass(frozen=True)
class BenchmarkSuite:
    """OOD protocol isolating structural game families from training mixtures."""

    actions: int
    batch_size: int = 1024

    def _distribution(self, family: int) -> ZeroSumGameDistribution:
        weights = [0.0, 0.0, 0.0, 0.0]
        weights[family] = 1.0
        return ZeroSumGameDistribution(self.actions, family_weights=tuple(weights))

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: str | torch.device) -> dict[str, dict[str, float]]:
        model.eval()
        results: dict[str, dict[str, float]] = {}
        for name, index in (("dense_antisymmetric", 0), ("low_rank", 1), ("transitive", 2), ("cyclic", 3)):
            game = self._distribution(index).sample(self.batch_size, device, difficulty=1.0)
            results[name] = summarize(model, game)
        return results
