from __future__ import annotations

import json
from pathlib import Path

from src.nuro_symbolic_regression.interfaces.cli import main


def test_benchmark_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark",
            "--benchmark",
            "mean",
            "--size",
            "60",
            "--population-size",
            "40",
            "--generations",
            "8",
            "--report-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0

    summary = tmp_path / "benchmark_summary.md"
    json_files = sorted(tmp_path.glob("*.json"))

    assert summary.exists()
    assert len(json_files) == 1

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["benchmark"] == "mean"
    assert "best_expression" in payload


def test_train_can_write_json_report(tmp_path: Path) -> None:
    exit_code = main(
        [
            "train",
            "--benchmark",
            "linear",
            "--size",
            "60",
            "--population-size",
            "40",
            "--generations",
            "8",
            "--save-report",
            "--report-dir",
            str(tmp_path),
            "--json",
        ]
    )

    assert exit_code == 0
    assert len(list(tmp_path.glob("*.json"))) == 1

