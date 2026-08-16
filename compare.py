from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bitminimax.baselines import MirrorProxOracle, UniformPolicy
from bitminimax.binary import BinaryMinimaxPolicy
from bitminimax.games import ZeroSumGameDistribution, exploitability
from bitminimax.league import CrossPlayLeague
from bitminimax.visualize import render_game_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BitMinimax baselines, cross-play, and visual diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--actions", type=int, default=7)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/comparison")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    model = BinaryMinimaxPolicy(args.actions, config["hidden_dim"], depth=config.get("depth", 3), heads=config.get("heads", 4)).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    game = ZeroSumGameDistribution(args.actions).sample(args.games, args.device)
    with torch.no_grad():
        row, col, _ = model(game)
        learned_nashconv = exploitability(game, row, col).mean().item()
    league = CrossPlayLeague(args.actions, games_per_match=args.games)
    standings = league.run({"BitMinimax": model, "MirrorProx oracle": MirrorProxOracle(), "Uniform": UniformPolicy()}, args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    render_game_diagnostic(game[:1], row[:1], col[:1], output / "game_diagnostic.png")
    report = {"bitminimax_nashconv": learned_nashconv, "standings": [entry.__dict__ for entry in standings], "diagnostic": str(output / "game_diagnostic.png")}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
