# BitMinimax

Binary neural policies for learning approximate Nash equilibria in procedurally generated zero-sum matrix games.

The project uses straight-through binary gates to constrain each player's strategy representation to `{-1, +1}`, while retaining gradients through a temperature-controlled continuous relaxation. It combines direct exploitability minimization with unrolled Mirror-Prox distillation, symmetry equivariance, adversarial payoff perturbations, and a game-complexity curriculum.

## Layout

- `bitminimax/games.py` — mixed game families, transformations, and equilibrium utilities
- `bitminimax/binary.py` — straight-through binary layers, attention, and policy network
- `bitminimax/solvers.py` — differentiable Mirror-Prox equilibrium targets
- `bitminimax/trainer.py` — curriculum self-play, robustness objectives, checkpointing
- `bitminimax/metrics.py` — exploitability, calibration, and binary-efficiency metrics
- `bitminimax/benchmarks.py` — OOD structural-generalization benchmark suite
- `bitminimax/deployment.py` — packed-bit BNN export and inference-cost estimation
- `bitminimax/baselines.py` — architecture-matched FP32, uniform, and Mirror-Prox controls
- `bitminimax/league.py` — solver cross-play tournament with Elo ranking
- `bitminimax/visualize.py` — payoff matrix and predicted-strategy plots
- `train.py` — training entry point
- `evaluate.py` — held-out exploitability report

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --actions 7 --epochs 800 --device cpu --solver-steps 32
```

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt --actions 7 --report reports/evaluation.json
```

Add `--export-binary exports/solver.pt` to produce a packed-bit deployment artifact.

## Reproducible comparison

```bash
python compare.py --checkpoint checkpoints/best.pt --actions 7
```

The comparison command runs BitMinimax against a uniform policy and a 256-step
Mirror-Prox reference, writes cross-play Elo standings to JSON, and renders a
diagnostic image of an unseen game with its learned mixed strategies.

## Quality checks

The repository includes invariant tests for anti-symmetric game generation,
simplex-valid policy outputs, and Mirror-Prox targets. GitHub Actions runs Ruff
and Pytest on every push and pull request.
