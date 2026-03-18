# Migration Notes

## Legacy files

The original experimental scripts are preserved under `legacy/`:

- `legacy/expressioncration.py`
- `legacy/neuro_symbol_network.py`
- `legacy/neuro_symbol_network3.py`
- `legacy/untitled87.py`
- `legacy/README.md`

They contain valuable experiments but were notebook exports with repeated class/function redefinitions.

## Canonical implementation

Use the package implementation under `src/nuro_symbolic_regression` for all new work.

## Suggested next migration step

Keep `legacy/` read-only and port remaining useful ideas (for example PyTorch constant tuning) into `infrastructure/` adapters with tests.

