from __future__ import annotations

import math

from ..domain.expression import ExpressionNode


def mse(expression: ExpressionNode, dataset: list[tuple[dict[str, float], float]]) -> float:
    errors = []
    for x, y in dataset:
        pred = expression.evaluate(x)
        err = (pred - y) ** 2
        if math.isnan(err) or math.isinf(err):
            err = 1e12
        errors.append(err)
    return sum(errors) / max(len(errors), 1)


def objective(
    expression: ExpressionNode,
    dataset: list[tuple[dict[str, float], float]],
    complexity_weight: float,
) -> float:
    return mse(expression, dataset) + complexity_weight * expression.size()

