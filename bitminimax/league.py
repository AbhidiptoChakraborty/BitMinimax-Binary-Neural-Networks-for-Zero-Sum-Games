from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from .games import ZeroSumGameDistribution, expected_payoff, exploitability


@dataclass(frozen=True)
class LeagueEntry:
    name: str
    elo: float
    mean_payoff: float
    exploitability: float


class CrossPlayLeague:
    """Evaluates solvers in cross-play instead of only independent test games.

    Each candidate supplies the row policy while every other candidate supplies
    the column policy on the same games. Elo provides a compact, familiar
    ranking while exploitability preserves the game-theoretic diagnostic.
    """

    def __init__(self, actions: int, games_per_match: int = 256, k_factor: float = 24.0) -> None:
        self.distribution = ZeroSumGameDistribution(actions)
        self.games_per_match, self.k_factor = games_per_match, k_factor

    @torch.no_grad()
    def run(self, solvers: Mapping[str, object], device: str | torch.device) -> list[LeagueEntry]:
        names, ratings = list(solvers), {name: 1000.0 for name in solvers}
        payoffs: dict[str, list[float]] = {name: [] for name in names}
        regrets: dict[str, list[float]] = {name: [] for name in names}
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                game = self.distribution.sample(self.games_per_match, device)
                first_row, first_col, _ = solvers[first](game)
                second_row, second_col, _ = solvers[second](game)
                first_score = expected_payoff(game, first_row, second_col).mean().item()
                second_score = expected_payoff(game, second_row, first_col).mean().item()
                outcome = 1.0 if first_score > second_score else 0.0 if first_score < second_score else 0.5
                expected = 1 / (1 + 10 ** ((ratings[second] - ratings[first]) / 400))
                delta = self.k_factor * (outcome - expected)
                ratings[first], ratings[second] = ratings[first] + delta, ratings[second] - delta
                payoffs[first].append(first_score)
                payoffs[second].append(second_score)
                regrets[first].append(exploitability(game, first_row, first_col).mean().item())
                regrets[second].append(exploitability(game, second_row, second_col).mean().item())
        return sorted((LeagueEntry(name, ratings[name], sum(payoffs[name]) / max(1, len(payoffs[name])), sum(regrets[name]) / max(1, len(regrets[name]))) for name in names), key=lambda entry: entry.elo, reverse=True)
