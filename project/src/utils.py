"""Shared utilities for the project."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_paths() -> dict[str, Path]:
    """Get common project paths."""
    return {
        "root": PROJECT_ROOT,
        "data_raw": PROJECT_ROOT / "data" / "raw",
        "data_processed": PROJECT_ROOT / "data" / "processed",
        "models": PROJECT_ROOT / "models",
        "results": PROJECT_ROOT / "results",
        "logs": PROJECT_ROOT / "experiments" / "logs",
        "notebooks": PROJECT_ROOT / "notebooks",
        "report": PROJECT_ROOT / "REPORT",
    }


def setup_logging(log_file: Path, script_name: str = "script") -> None:
    """Setup logging configuration."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("Logging initialized for %s", script_name)

