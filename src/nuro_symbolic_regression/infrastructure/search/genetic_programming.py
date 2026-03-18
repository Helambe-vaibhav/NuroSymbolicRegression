from __future__ import annotations

import random
from collections.abc import Callable

from ...application.config import EvolutionConfig
from ...domain.expression import ExpressionNode
from ...domain.operators import BINARY_OPERATORS, DEFAULT_OPERATOR_SET, UNARY_OPERATORS


class GeneticProgrammingSearch:
    """Genetic programming engine for symbolic regression trees."""

    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config
        self.rng = random.Random(config.random_seed)

    def _random_constant(self) -> ExpressionNode:
        value = round(self.rng.uniform(self.config.constant_low, self.config.constant_high), 3)
        return ExpressionNode(value)

    def _random_terminal(self) -> ExpressionNode:
        if self.rng.random() < 0.65:
            return ExpressionNode(self.rng.choice(self.config.variables))
        return self._random_constant()

    def _random_tree(self, depth: int = 0) -> ExpressionNode:
        if depth >= self.config.max_depth or (depth > 0 and self.rng.random() < 0.25):
            return self._random_terminal()

        if self.rng.random() < 0.35:
            op = self.rng.choice(UNARY_OPERATORS)
            return ExpressionNode(op, left=self._random_tree(depth + 1))

        op = self.rng.choice(BINARY_OPERATORS)
        return ExpressionNode(op, left=self._random_tree(depth + 1), right=self._random_tree(depth + 1))

    def _evaluate_population(
        self,
        population: list[ExpressionNode],
        score_fn: Callable[[ExpressionNode], float],
    ) -> list[tuple[ExpressionNode, float]]:
        scored = [(ind, score_fn(ind)) for ind in population]
        scored.sort(key=lambda pair: pair[1])
        return scored

    def _tournament_select(self, scored: list[tuple[ExpressionNode, float]]) -> ExpressionNode:
        choices = [scored[self.rng.randrange(0, len(scored))] for _ in range(self.config.tournament_size)]
        return min(choices, key=lambda pair: pair[1])[0]

    def _mutate(self, expr: ExpressionNode) -> ExpressionNode:
        if self.rng.random() >= self.config.mutation_rate:
            return expr.clone()

        paths = expr.paths()
        path = self.rng.choice(paths)
        new_subtree = self._random_tree(depth=min(len(path), self.config.max_depth))
        return expr.replace(path, new_subtree)

    def _crossover(self, left_parent: ExpressionNode, right_parent: ExpressionNode) -> ExpressionNode:
        left_paths = left_parent.paths()
        right_paths = right_parent.paths()

        left_path = self.rng.choice(left_paths)
        right_path = self.rng.choice(right_paths)
        right_subtree = right_parent.node_at(right_path)

        child = left_parent.replace(left_path, right_subtree)
        return child

    def evolve(
        self,
        score_fn: Callable[[ExpressionNode], float],
    ) -> tuple[ExpressionNode, float, list[float]]:
        population = [self._random_tree() for _ in range(self.config.population_size)]
        best_score = float("inf")
        best_expr = population[0].clone()
        history: list[float] = []

        for _ in range(self.config.generations):
            scored = self._evaluate_population(population, score_fn)
            gen_best_expr, gen_best_score = scored[0]
            history.append(gen_best_score)

            if gen_best_score < best_score:
                best_score = gen_best_score
                best_expr = gen_best_expr.clone()

            elites = [expr.clone() for expr, _ in scored[: self.config.elitism]]
            next_population = elites

            while len(next_population) < self.config.population_size:
                if self.rng.random() < self.config.crossover_rate:
                    p1 = self._tournament_select(scored)
                    p2 = self._tournament_select(scored)
                    child = self._crossover(p1, p2)
                else:
                    p1 = self._tournament_select(scored)
                    child = p1.clone()
                child = self._mutate(child)
                next_population.append(child)

            population = next_population

        return best_expr, best_score, history


def supported_operators() -> tuple[str, ...]:
    return tuple(DEFAULT_OPERATOR_SET.keys())

