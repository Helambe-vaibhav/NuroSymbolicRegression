from __future__ import annotations

import math
import random
from collections.abc import Callable


def benchmark_registry() -> dict[str, Callable[[dict[str, float]], float]]:
    return {
        "mean": lambda x: (x["x0"] + x["x1"]) / 2.0,
        "linear": lambda x: 3.0 * x["x0"] - 2.0 * x["x1"] + 1.0,
        "product": lambda x: x["x0"] * x["x1"],
        "poly": lambda x: x["x0"] ** 2 + 2.0 * x["x1"] + 0.5,
        "sine_mix": lambda x: math.sin(x["x0"]) + math.cos(x["x1"]),
    }


def generate_dataset(
    benchmark_name: str,
    size: int,
    seed: int = 7,
    low: float = -3.0,
    high: float = 3.0,
) -> list[tuple[dict[str, float], float]]:
    funcs = benchmark_registry()
    if benchmark_name not in funcs:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    rng = random.Random(seed)
    fn = funcs[benchmark_name]
    data: list[tuple[dict[str, float], float]] = []
    for _ in range(size):
        x = {
            "x0": rng.uniform(low, high),
            "x1": rng.uniform(low, high),
        }
        data.append((x, fn(x)))
    return data

