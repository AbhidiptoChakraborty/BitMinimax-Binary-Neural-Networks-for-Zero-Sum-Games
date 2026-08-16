from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack a flat uint8 {0,1} tensor into bytes without external libraries."""
    padding = (-bits.numel()) % 8
    if padding:
        bits = torch.cat((bits, torch.zeros(padding, dtype=torch.uint8)))
    groups = bits.reshape(-1, 8).to(torch.int16)
    shifts = torch.arange(8, dtype=torch.int16)
    return (groups * (1 << shifts)).sum(dim=1).to(torch.uint8)


@torch.no_grad()
def export_packed_binary_model(model: nn.Module, destination: str | Path) -> dict[str, float | int]:
    """Export sign-binarized weight tensors in packed-bit form.

    This is intentionally framework-light: the artifact can be consumed by a
    custom XNOR-popcount kernel, FPGA flow, or embedded inference runtime.
    Normalization/bias terms remain float32, which is standard for practical
    BNN deployments.
    """
    packed: dict[str, torch.Tensor] = {}
    float_tensors: dict[str, torch.Tensor] = {}
    binary_values = float_values = 0
    for name, tensor in model.state_dict().items():
        is_binary_weight = name.endswith("weight") and tensor.ndim == 2
        if is_binary_weight:
            bits = (tensor >= 0).to(torch.uint8).flatten().cpu()
            packed[name] = _pack_bits(bits)
            binary_values += tensor.numel()
        else:
            float_tensors[name] = tensor.cpu()
            float_values += tensor.numel()
    artifact = {"format": "bitminimax-packed-v1", "packed_binary_weights": packed, "float_tensors": float_tensors}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    fp32_bytes = (binary_values + float_values) * 4
    packed_bytes = sum(t.numel() for t in packed.values()) + float_values * 4
    return {
        "binary_values": binary_values,
        "float_values": float_values,
        "fp32_bytes": fp32_bytes,
        "packed_bytes": packed_bytes,
        "compression_ratio": fp32_bytes / max(packed_bytes, 1),
    }


@torch.no_grad()
def estimate_bit_operations(model: nn.Module, actions: int) -> int:
    """Approximate binary multiply-accumulates for one matrix-game inference."""
    total = 0
    for module in model.modules():
        if module.__class__.__name__ == "BinaryLinear":
            total += module.weight.numel()
    # Attention produces quadratic token interactions after binary projections.
    attention_cost = sum(actions * actions * getattr(m, "head_dim", 0) * getattr(m, "heads", 0) for m in model.modules())
    return total + attention_cost
