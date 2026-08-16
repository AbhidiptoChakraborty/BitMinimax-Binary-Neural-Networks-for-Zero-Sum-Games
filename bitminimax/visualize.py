from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .games import expected_payoff, exploitability


@torch.no_grad()
def render_game_diagnostic(game: Tensor, row: Tensor, col: Tensor, output: str | Path, title: str = "BitMinimax game diagnostic") -> None:
    """Create a portfolio-ready payoff/strategy diagnostic for a single game."""
    if game.ndim == 3:
        game, row, col = game[0], row[0], col[0]
    payoff = expected_payoff(game.unsqueeze(0), row.unsqueeze(0), col.unsqueeze(0)).item()
    regret = exploitability(game.unsqueeze(0), row.unsqueeze(0), col.unsqueeze(0)).item()
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.45, 1, 1]})
    image = axes[0].imshow(game.cpu(), cmap="coolwarm", vmin=-game.abs().max().item(), vmax=game.abs().max().item())
    axes[0].set_title("Payoff matrix A")
    axes[0].set_xlabel("Column action")
    axes[0].set_ylabel("Row action")
    figure.colorbar(image, ax=axes[0], shrink=0.82)
    actions = torch.arange(row.numel()).cpu()
    axes[1].bar(actions - 0.18, row.cpu(), width=0.36, label="row", color="#2E86AB")
    axes[1].bar(actions + 0.18, col.cpu(), width=0.36, label="column", color="#E07A5F")
    axes[1].set_title("Predicted mixed strategies")
    axes[1].set_xlabel("Action")
    axes[1].set_ylabel("Probability")
    axes[1].legend(frameon=False)
    axes[2].axis("off")
    axes[2].text(0.05, 0.72, f"Expected payoff\n{payoff:+.4f}", fontsize=16, weight="bold")
    axes[2].text(0.05, 0.38, f"NashConv\n{regret:.4f}", fontsize=16, weight="bold")
    axes[2].text(0.05, 0.11, "Lower NashConv means\na less exploitable strategy.", fontsize=10)
    figure.suptitle(title, weight="bold")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
