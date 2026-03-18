from __future__ import annotations

from dataclasses import dataclass

from .config import EvolutionConfig
from .fitness import objective
from ..domain.expression import ExpressionNode
from ..infrastructure.search.genetic_programming import GeneticProgrammingSearch


@dataclass(frozen=True)
class TrainingResult:
    best_expression: ExpressionNode
    best_score: float
    history: list[float]


def train_symbolic_regressor(
    dataset: list[tuple[dict[str, float], float]],
    config: EvolutionConfig,
) -> TrainingResult:
    search = GeneticProgrammingSearch(config)

    def score_fn(expr: ExpressionNode) -> float:
        return objective(expr, dataset, config.complexity_weight)

    best_expr, best_score, history = search.evolve(score_fn)
    return TrainingResult(best_expression=best_expr, best_score=best_score, history=history)

