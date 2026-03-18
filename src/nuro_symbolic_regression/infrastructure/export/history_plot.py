from __future__ import annotations

from pathlib import Path


def save_history_plot(history: list[float], output_path: str | Path, title: str) -> Path | None:
    """Save a convergence plot when matplotlib is available.

    Returns the created path or None when plotting is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(history, color="#1f77b4", linewidth=1.8)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Objective")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path

