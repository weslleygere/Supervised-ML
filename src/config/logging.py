"""Run-local logging, with no shared temporary file between experiments."""
import logging
from pathlib import Path


def configure_logging(output_dir: Path, level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "experiment.log")],
        force=True,
    )
    logging.captureWarnings(True)
