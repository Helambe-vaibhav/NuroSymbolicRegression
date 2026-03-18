from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..application.config import EvolutionConfig
from ..application.use_cases import train_symbolic_regressor
from ..domain.benchmarks import benchmark_registry, generate_dataset
from ..infrastructure.export.benchmark_report import (
    BenchmarkRunRecord,
    ensure_report_dir,
    write_run_json,
    write_summary_markdown,
)
from ..infrastructure.export.history_plot import save_history_plot
from ..infrastructure.export.sympy_exporter import to_latex, to_sympy


def _build_config(args: argparse.Namespace) -> EvolutionConfig:
    return EvolutionConfig(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        max_depth=args.max_depth,
        random_seed=args.seed,
        complexity_weight=args.complexity_weight,
    )


def cmd_train(args: argparse.Namespace) -> int:
    config = _build_config(args)
    dataset = generate_dataset(args.benchmark, args.size, seed=args.seed)
    result = train_symbolic_regressor(dataset, config)

    payload = {
        "benchmark": args.benchmark,
        "config": asdict(config),
        "best_score": result.best_score,
        "best_expression": result.best_expression.to_infix(),
        "sympy": str(to_sympy(result.best_expression)),
        "latex": to_latex(result.best_expression),
        "history_tail": result.history[-10:],
    }

    report_dir = ensure_report_dir(args.report_dir)

    if args.save_report:
        record = BenchmarkRunRecord(
            benchmark=args.benchmark,
            best_score=result.best_score,
            best_expression=result.best_expression.to_infix(),
            sympy=str(to_sympy(result.best_expression)),
            latex=to_latex(result.best_expression),
            generations=config.generations,
            population_size=config.population_size,
            seed=config.random_seed,
            history=result.history,
        )
        report_path = write_run_json(record, report_dir)
        payload["report_path"] = str(report_path)

    if args.plot_history:
        plot_path = report_dir / f"train_{args.benchmark}_history.png"
        created = save_history_plot(result.history, plot_path, f"Convergence - {args.benchmark}")
        payload["plot_path"] = str(created) if created is not None else None

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Benchmark: {payload['benchmark']}")
        print(f"Best score: {payload['best_score']:.6f}")
        print(f"Best expression: {payload['best_expression']}")
        print(f"SymPy: {payload['sympy']}")
        print(f"LaTeX: {payload['latex']}")
        if args.save_report:
            print(f"Report: {payload['report_path']}")
        if args.plot_history:
            if payload["plot_path"] is None:
                print("Plot: skipped (matplotlib not installed)")
            else:
                print(f"Plot: {payload['plot_path']}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = _build_config(args)
    names = list(benchmark_registry().keys()) if args.all else [args.benchmark]
    report_dir = ensure_report_dir(args.report_dir)
    records: list[BenchmarkRunRecord] = []

    for name in names:
        dataset = generate_dataset(name, args.size, seed=args.seed)
        result = train_symbolic_regressor(dataset, config)

        record = BenchmarkRunRecord(
            benchmark=name,
            best_score=result.best_score,
            best_expression=result.best_expression.to_infix(),
            sympy=str(to_sympy(result.best_expression)),
            latex=to_latex(result.best_expression),
            generations=config.generations,
            population_size=config.population_size,
            seed=config.random_seed,
            history=result.history,
        )
        records.append(record)

        if not args.no_report:
            write_run_json(record, report_dir)

        if args.plot_history:
            plot_path = report_dir / f"benchmark_{name}_history.png"
            save_history_plot(result.history, plot_path, f"Convergence - {name}")

        print(f"[{name}] score={result.best_score:.6f} expr={result.best_expression.to_infix()}")

    if not args.no_report and records:
        summary_path = write_summary_markdown(records, report_dir)
        print(f"Summary: {summary_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsr",
        description="Nuro Symbolic Regression CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--benchmark", default="mean", choices=tuple(benchmark_registry().keys()))
    common.add_argument("--size", type=int, default=300)
    common.add_argument("--population-size", type=int, default=160)
    common.add_argument("--generations", type=int, default=80)
    common.add_argument("--mutation-rate", type=float, default=0.25)
    common.add_argument("--crossover-rate", type=float, default=0.70)
    common.add_argument("--complexity-weight", type=float, default=0.001)
    common.add_argument("--max-depth", type=int, default=5)
    common.add_argument("--seed", type=int, default=7)
    common.add_argument("--report-dir", type=Path, default=Path("results"))

    train = sub.add_parser("train", parents=[common], help="Train on a benchmark function")
    train.add_argument("--json", action="store_true", help="Print JSON output")
    train.add_argument("--save-report", action="store_true", help="Write JSON report to --report-dir")
    train.add_argument("--plot-history", action="store_true", help="Save convergence chart if matplotlib is available")
    train.set_defaults(handler=cmd_train)

    bench = sub.add_parser("benchmark", parents=[common], help="Run one or all benchmarks")
    bench.add_argument("--all", action="store_true", help="Run all built-in benchmarks")
    bench.add_argument("--no-report", action="store_true", help="Do not write results/*.json and summary markdown")
    bench.add_argument("--plot-history", action="store_true", help="Save convergence charts if matplotlib is available")
    bench.set_defaults(handler=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

