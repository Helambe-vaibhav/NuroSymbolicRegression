from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRunRecord:
    benchmark: str
    best_score: float
    best_expression: str
    sympy: str
    latex: str
    generations: int
    population_size: int
    seed: int
    history: list[float]


def ensure_report_dir(report_dir: str | Path) -> Path:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").lower()


def write_run_json(record: BenchmarkRunRecord, report_dir: str | Path) -> Path:
    directory = ensure_report_dir(report_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_name = f"{_safe_name(record.benchmark)}_{record.seed}_{timestamp}.json"
    out_path = directory / file_name
    out_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return out_path


def build_markdown_table(records: list[BenchmarkRunRecord]) -> str:
    header = "| Benchmark | Best Score | Best Expression |\n|---|---:|---|"
    rows = [
        f"| {r.benchmark} | {r.best_score:.6f} | `{r.best_expression}` |"
        for r in records
    ]
    return "\n".join([header, *rows])


def write_summary_markdown(records: list[BenchmarkRunRecord], report_dir: str | Path) -> Path:
    directory = ensure_report_dir(report_dir)
    out_path = directory / "benchmark_summary.md"
    content = "# Benchmark Summary\n\n" + build_markdown_table(records) + "\n"
    out_path.write_text(content, encoding="utf-8")
    return out_path

