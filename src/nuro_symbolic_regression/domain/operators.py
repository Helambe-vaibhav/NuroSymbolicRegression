from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class OperatorSpec:
    symbol: str
    arity: int
    func: Callable[..., float]


def _safe_div(x: float, y: float) -> float:
    return x / y if abs(y) > 1e-8 else 1.0


def _safe_pow(x: float, y: float) -> float:
    # Clamp exponent and use abs base to avoid complex values.
    return math.pow(abs(x), max(min(y, 8.0), -8.0))


def _safe_exp(x: float) -> float:
    return math.exp(max(min(x, 20.0), -20.0))


def _safe_log(x: float) -> float:
    return math.log1p(abs(x))


DEFAULT_OPERATOR_SET: dict[str, OperatorSpec] = {
    "+": OperatorSpec("+", 2, lambda a, b: a + b),
    "-": OperatorSpec("-", 2, lambda a, b: a - b),
    "*": OperatorSpec("*", 2, lambda a, b: a * b),
    "/": OperatorSpec("/", 2, _safe_div),
    "^": OperatorSpec("^", 2, _safe_pow),
    "sin": OperatorSpec("sin", 1, math.sin),
    "cos": OperatorSpec("cos", 1, math.cos),
    "exp": OperatorSpec("exp", 1, _safe_exp),
    "log": OperatorSpec("log", 1, _safe_log),
}

OP_ARITY: dict[str, int] = {name: spec.arity for name, spec in DEFAULT_OPERATOR_SET.items()}
OP_FUNC: dict[str, Callable[..., float]] = {name: spec.func for name, spec in DEFAULT_OPERATOR_SET.items()}


BINARY_OPERATORS = tuple(op for op, spec in DEFAULT_OPERATOR_SET.items() if spec.arity == 2)
UNARY_OPERATORS = tuple(op for op, spec in DEFAULT_OPERATOR_SET.items() if spec.arity == 1)

