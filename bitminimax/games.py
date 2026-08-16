from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class ZeroSumGameDistribution:
    """Procedurally generates normalized anti-symmetric zero-sum games.

    The row player receives payoff ``x^T A y``; the column player receives its
    negative. Anti-symmetry removes arbitrary payoff drift and creates a rich
    family of cyclic, non-transitive games beyond rock-paper-scissors.
    """

    actions: int
    payoff_scale: float = 1.0
    family_weights: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)

    def sample(self, batch_size: int, device: torch.device | str, difficulty: float = 1.0) -> Tensor:
        """Sample cyclic, transitive, low-rank, and dense games.

        Difficulty governs the amplitude of non-transitive components; it is
        intended for curriculum training from smooth to adversarial landscapes.
        """
        device = torch.device(device)
        n = self.actions
        family = torch.multinomial(torch.tensor(self.family_weights, device=device), batch_size, replacement=True)
        raw = torch.randn(batch_size, self.actions, self.actions, device=device)
        dense = raw - raw.transpose(-1, -2)
        ranks = torch.randn(batch_size, n, 1, device=device)
        low_rank = ranks - ranks.transpose(-1, -2)
        strength = torch.linspace(-1, 1, n, device=device).expand(batch_size, -1)
        transitive = strength.unsqueeze(-1) - strength.unsqueeze(-2)
        offsets = torch.arange(n, device=device)
        distance = (offsets[None, :] - offsets[:, None]) % n
        cyclic = torch.where(distance <= n // 2, torch.ones_like(distance), -torch.ones_like(distance)).float()
        cyclic.fill_diagonal_(0)
        bank = torch.stack((dense, low_rank, transitive, cyclic.expand(batch_size, -1, -1)), dim=1)
        game = bank[torch.arange(batch_size, device=device), family]
        game = game + difficulty * 0.2 * dense
        game = game / game.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        return game * self.payoff_scale


def transform_game(game: Tensor, permutation: Tensor | None = None, noise_scale: float = 0.0) -> Tensor:
    """Apply a payoff-preserving action relabeling plus anti-symmetric noise."""
    if permutation is not None:
        game = game[:, permutation][:, :, permutation]
    if noise_scale:
        noise = torch.randn_like(game)
        game = game + noise_scale * (noise - noise.transpose(-1, -2))
    return game


def expected_payoff(game: Tensor, row_strategy: Tensor, col_strategy: Tensor) -> Tensor:
    """Return batched row-player payoff for mixed strategies."""
    return torch.einsum("bi,bij,bj->b", row_strategy, game, col_strategy)


def pure_best_response_payoffs(game: Tensor, row_strategy: Tensor, col_strategy: Tensor) -> tuple[Tensor, Tensor]:
    """Return row's max payoff and column's minimum row payoff against policies."""
    row_values = torch.einsum("bij,bj->bi", game, col_strategy)
    col_values = torch.einsum("bi,bij->bj", row_strategy, game)
    return row_values.max(dim=-1).values, col_values.min(dim=-1).values


def exploitability(game: Tensor, row_strategy: Tensor, col_strategy: Tensor) -> Tensor:
    """NashConv for a two-player zero-sum game (zero at equilibrium)."""
    row_br, col_br = pure_best_response_payoffs(game, row_strategy, col_strategy)
    return row_br - col_br
