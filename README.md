# Nuro Symbolic Regression

A genetic programming–based symbolic regression engine that discovers interpretable mathematical expressions from data. The system evolves expression trees using mutation and crossover, optimizing a complexity-aware objective that balances prediction accuracy and model simplicity. It includes a modular architecture, benchmarking suite, and CLI for reproducible experiments and analysis.

## What this project does

- Evolves symbolic expressions with genetic programming.
- Supports arithmetic and transcendental operators with safe evaluation.
- Trains on built-in benchmark functions (`mean`, `linear`, `product`, `poly`, `sine_mix`).
- Exports final expressions to SymPy and LaTeX.

## Architecture

- `src/nuro_symbolic_regression/domain`: expression trees, operator semantics, benchmark definitions.
- `src/nuro_symbolic_regression/application`: config, fitness objective, training use case.
- `src/nuro_symbolic_regression/infrastructure`: GP search engine and exporters.
- `src/nuro_symbolic_regression/interfaces`: command-line interface.
- `tests`: unit and smoke tests.
- `docs/architecture.md`: blueprint.
- `docs/migration.md`: migration mapping from legacy scripts.

## Quick start

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

## CLI workflow

### Train one benchmark

```powershell
nsr train --benchmark mean --size 300 --generations 80 --population-size 160
```

### Train and return JSON payload

```powershell
nsr train --benchmark linear --json
```

### Run benchmark sweep

```powershell
nsr benchmark --all --generations 60 --population-size 120
```

### Run benchmark sweep and generate reports

```powershell
nsr benchmark --all --report-dir results
```

This writes per-benchmark JSON files to `results/*.json` and a markdown table summary at `results/benchmark_summary.md`.

### Optional convergence plots

```powershell
nsr train --benchmark mean --plot-history --report-dir results
nsr benchmark --all --plot-history --report-dir results
```

If `matplotlib` is installed, history plots are saved into the report directory.

## Python API usage

```python
from nuro_symbolic_regression.application.config import EvolutionConfig
from nuro_symbolic_regression.application.use_cases import train_symbolic_regressor
from nuro_symbolic_regression.domain.benchmarks import generate_dataset

config = EvolutionConfig(generations=40, population_size=100, random_seed=7)
data = generate_dataset("mean", size=300, seed=7)
result = train_symbolic_regressor(data, config)

print(result.best_expression.to_infix())
print(result.best_score)
```

## Recommended workflow

1. Add or adjust benchmark in `domain/benchmarks.py`.
2. Tune `EvolutionConfig` in `application/config.py`.
3. Run `pytest`.
4. Run CLI benchmark sweep and collect best expressions.
5. Commit with benchmark output snapshots for portfolio evidence.

#   N u r o S y m b o l i c R e g r e s s i o n 
 
 
