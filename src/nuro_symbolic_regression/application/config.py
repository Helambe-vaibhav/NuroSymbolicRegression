from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionConfig:
    variables: tuple[str, ...] = ("x0", "x1")
    max_depth: int = 5
    population_size: int = 160
    generations: int = 80
    tournament_size: int = 5
    elitism: int = 8
    mutation_rate: float = 0.25
    crossover_rate: float = 0.70
    complexity_weight: float = 0.001
    constant_low: float = -5.0
    constant_high: float = 5.0
    random_seed: int = 7

