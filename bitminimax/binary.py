from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StraightThroughSign(torch.autograd.Function):
    """Hard sign in forward pass, clipped identity surrogate in backward pass."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return x.sign().masked_fill(x == 0, 1.0)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).to(grad_output.dtype)


def binary_sign(x: Tensor) -> Tensor:
    return StraightThroughSign.apply(x)


class BinaryLinear(nn.Module):
    """Fully binary affine layer with learned full-precision latent weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        binary_weight = binary_sign(self.weight)
        return F.linear(x, binary_weight, self.bias)


class BinaryAttentionBlock(nn.Module):
    """Binary projections with full-precision normalization and residual routing."""

    def __init__(self, dimension: int, heads: int = 4, expansion: int = 2) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.heads, self.head_dim = heads, dimension // heads
        self.qkv = BinaryLinear(dimension, 3 * dimension, bias=False)
        self.projection = BinaryLinear(dimension, dimension, bias=False)
        self.feedforward = nn.Sequential(BinaryLinear(dimension, dimension * expansion), nn.Hardtanh(), BinaryLinear(dimension * expansion, dimension))
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dimension = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        def heads(t: Tensor) -> Tensor:
            return t.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        attention = (q @ k.transpose(-2, -1) / self.head_dim**0.5).softmax(-1)
        attended = (attention @ v).transpose(1, 2).reshape(batch, tokens, dimension)
        x = x + self.projection(attended)
        return x + self.feedforward(self.norm2(x))


class BinaryMinimaxPolicy(nn.Module):
    """Maps a payoff matrix to both players' mixed strategies via binary gates."""

    def __init__(self, actions: int, hidden_dim: int = 192, temperature: float = 0.7, depth: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.actions = actions
        self.temperature = temperature
        self.action_embedding = nn.Parameter(torch.randn(actions, hidden_dim) * 0.02)
        self.row_projection = BinaryLinear(actions, hidden_dim)
        self.encoder = nn.ModuleList([BinaryAttentionBlock(hidden_dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.row_head = BinaryLinear(hidden_dim, actions)
        self.col_head = BinaryLinear(hidden_dim, actions)

    def forward(self, game: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        # Each action is a token containing its payoff profile against all actions.
        x = self.row_projection(game) + self.action_embedding.unsqueeze(0)
        for block in self.encoder:
            x = block(x)
        features = self.final_norm(x).mean(dim=1)
        row_logits = self.row_head(features)
        col_logits = self.col_head(features)
        row = F.softmax(row_logits / self.temperature, dim=-1)
        col = F.softmax(col_logits / self.temperature, dim=-1)
        diagnostics = {"row_logits": row_logits, "col_logits": col_logits, "binary_features": binary_sign(features)}
        return row, col, diagnostics
