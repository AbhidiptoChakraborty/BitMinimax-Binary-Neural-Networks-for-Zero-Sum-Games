import torch

from bitminimax.binary import BinaryMinimaxPolicy
from bitminimax.games import ZeroSumGameDistribution, exploitability
from bitminimax.solvers import mirror_prox


def test_sampled_games_are_antisymmetric() -> None:
    game = ZeroSumGameDistribution(actions=5).sample(batch_size=3, device="cpu")
    assert torch.allclose(game, -game.transpose(-1, -2), atol=1e-6)


def test_policy_outputs_valid_mixed_strategies() -> None:
    game = ZeroSumGameDistribution(actions=5).sample(batch_size=4, device="cpu")
    row, col, _ = BinaryMinimaxPolicy(actions=5, hidden_dim=20, heads=4)(game)
    assert row.shape == col.shape == (4, 5)
    assert torch.allclose(row.sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(col.sum(-1), torch.ones(4), atol=1e-6)


def test_mirror_prox_returns_simplex_strategies() -> None:
    game = ZeroSumGameDistribution(actions=5).sample(batch_size=4, device="cpu")
    row, col = mirror_prox(game, steps=4)
    assert torch.allclose(row.sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(col.sum(-1), torch.ones(4), atol=1e-6)
    assert exploitability(game, row, col).isfinite().all()
