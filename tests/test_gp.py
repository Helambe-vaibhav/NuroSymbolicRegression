from src.nuro_symbolic_regression.application.config import EvolutionConfig
from src.nuro_symbolic_regression.application.use_cases import train_symbolic_regressor
from src.nuro_symbolic_regression.domain.benchmarks import generate_dataset


def test_gp_training_smoke() -> None:
    config = EvolutionConfig(
        population_size=80,
        generations=20,
        max_depth=4,
        random_seed=11,
    )
    data = generate_dataset("mean", size=120, seed=11)
    result = train_symbolic_regressor(data, config)

    assert len(result.history) == config.generations
    assert result.best_score == min(result.history)
    assert result.best_score < 20.0

    pred = result.best_expression.evaluate({"x0": 1.0, "x1": 2.0})
    assert abs(pred) < 1e10

from src.nuro_symbolic_regression.application.config import EvolutionConfig
from src.nuro_symbolic_regression.application.use_cases import train_symbolic_regressor
from src.nuro_symbolic_regression.domain.benchmarks import generate_dataset

config = EvolutionConfig(generations=50, population_size=300, random_seed=7)
data = generate_dataset("mean", size=500, seed=7)
print(data[:5])
result = train_symbolic_regressor(data, config)

print(result.best_expression.to_infix())
print(result.best_score)