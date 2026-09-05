from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bitminimax.benchmarks import BenchmarkSuite
from bitminimax.binary import BinaryMinimaxPolicy
from bitminimax.deployment import estimate_bit_operations, export_packed_binary_model
from bitminimax.games import ZeroSumGameDistribution, expected_payoff, exploitability
from bitminimax.metrics import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a binary minimax policy on unseen games.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--actions", type=int, default=7)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default=None, help="Optional JSON output path")
    parser.add_argument("--export-binary", default=None, help="Optional packed-weight output path")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    model = BinaryMinimaxPolicy(args.actions, config["hidden_dim"], depth=config.get("depth", 3), heads=config.get("heads", 4)).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    game = ZeroSumGameDistribution(args.actions).sample(args.games, args.device)
    with torch.no_grad():
        row, col, _ = model(game)
        report = {
            "mean_exploitability": exploitability(game, row, col).mean().item(),
            "p95_exploitability": torch.quantile(exploitability(game, row, col), 0.95).item(),
            "mean_abs_payoff": expected_payoff(game, row, col).abs().mean().item(),
            "mean_strategy_entropy": (-(row * row.clamp_min(1e-8).log()).sum(-1)).mean().item(),
        }
        report.update({f"diagnostic_{key}": value for key, value in summarize(model, game).items()})
        report["ood_suite"] = BenchmarkSuite(args.actions).evaluate(model, args.device)
        report["estimated_bit_operations"] = estimate_bit_operations(model, args.actions)
    if args.export_binary:
        report["binary_export"] = export_packed_binary_model(model, args.export_binary)
    for name, value in report.items():
        print(f"{name}: {value:.6f}" if isinstance(value, float) else f"{name}: {value}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
