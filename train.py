from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import trange

from bitminimax.binary import BinaryMinimaxPolicy
from bitminimax.benchmarks import BenchmarkSuite
from bitminimax.games import ZeroSumGameDistribution
from bitminimax.trainer import MinimaxTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary neural minimax solver.")
    parser.add_argument("--actions", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--solver-steps", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="checkpoints/best.pt")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--benchmark-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = TrainConfig(actions=args.actions, hidden_dim=args.hidden_dim, depth=args.depth, heads=args.heads, batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate, solver_steps=args.solver_steps, device=args.device)
    model = BinaryMinimaxPolicy(config.actions, config.hidden_dim, depth=config.depth, heads=config.heads).to(config.device)
    trainer = MinimaxTrainer(model, ZeroSumGameDistribution(config.actions), config)
    best = float("inf")
    for epoch in trange(1, config.epochs + 1, desc="training"):
        train_metrics = trainer.train_epoch(epoch)
        if epoch % 10 == 0 or epoch == config.epochs:
            metrics = trainer.evaluate()
            if metrics["exploitability"] < best:
                best = metrics["exploitability"]
                trainer.checkpoint(args.output, epoch, metrics)
            print(f"epoch={epoch:04d} train_loss={train_metrics['loss']:.4f} val_nashconv={metrics['exploitability']:.4f} symmetry={metrics['symmetry_error']:.4f}")
        if epoch % args.benchmark_every == 0:
            suite = BenchmarkSuite(config.actions, batch_size=min(1024, config.batch_size * 2))
            ood = suite.evaluate(model, config.device)
            print("ood_nashconv=" + ", ".join(f"{name}:{score['exploitability']:.4f}" for name, score in ood.items()))
    print(f"best checkpoint: {Path(args.output)} | exploitability={best:.5f}")


if __name__ == "__main__":
    main()
