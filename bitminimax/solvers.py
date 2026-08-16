from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def mirror_prox(game: Tensor, steps: int = 32, step_size: float = 0.35) -> tuple[Tensor, Tensor]:
    """Unrolled entropic Mirror-Prox for batched zero-sum matrix games.

    Averaged iterates converge to a saddle point and provide stable soft targets
    without relying on an external linear-programming dependency.
    """
    batch, actions, _ = game.shape
    row_logits = torch.zeros(batch, actions, device=game.device, dtype=game.dtype)
    col_logits = torch.zeros_like(row_logits)
    row_average = torch.zeros_like(row_logits)
    col_average = torch.zeros_like(col_logits)
    for _ in range(steps):
        row, col = F.softmax(row_logits, -1), F.softmax(col_logits, -1)
        row_mid = F.softmax(row_logits + step_size * torch.einsum("bij,bj->bi", game, col), -1)
        col_mid = F.softmax(col_logits - step_size * torch.einsum("bi,bij->bj", row, game), -1)
        row_logits = row_logits + step_size * torch.einsum("bij,bj->bi", game, col_mid)
        col_logits = col_logits - step_size * torch.einsum("bi,bij->bj", row_mid, game)
        row_average += row_mid
        col_average += col_mid
    return row_average.div(steps), col_average.div(steps)
