# Clean Architecture Blueprint

This repository now follows a strict layer split so algorithm work does not leak into CLI or benchmark concerns.

## Layers

- `domain/`: pure expression model, operators, benchmark functions.
- `application/`: use-case orchestration and fitness objective.
- `infrastructure/`: search strategy (genetic programming) and exporters.
- `interfaces/`: CLI entrypoint (`nsr`) for train and benchmark workflows.

## Main flow

1. CLI parses runtime settings into `EvolutionConfig`.
2. Domain builds benchmark dataset.
3. Application use case runs `train_symbolic_regressor`.
4. Infrastructure GP engine evolves expressions.
5. Exporter prints SymPy and LaTeX form for portfolio output.
6. Optional exporters write `results/*.json`, markdown summary tables, and convergence charts.

## Why this is portfolio-grade

- Deterministic runs via seeds.
- Unit tests for expression and GP smoke checks.
- CI workflow validates installation + tests.
- Legacy notebook-style scripts are left intact, but isolated from the package.

