"""Atomic run artifacts and strict experiment identity checks."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import joblib


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, default=json_default)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default))
    os.replace(temporary, path)


def write_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    else:
        frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def write_bundle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash():
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for path in sorted([*root.joinpath("src").rglob("*.py"), root / "main.py", root / "requirements.txt"]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def identity(settings):
    from src.core.models.search_space import SEARCH_SPACE_VERSION
    values = settings.to_dict()
    runtime = {"output_dir", "run_mode", "source_run", "resume_run", "log_level", "effort_analysis_counts", "effort_analysis_repeats", "bootstrap_repeats", "confidence_level", "plateau_tolerance", "plateau_window"}
    result = {"settings": {k: v for k, v in values.items() if k not in runtime},
              "data_hash": file_hash(settings.data_path), "schema_hash": file_hash(settings.schema_path),
              "code_hash": code_hash(), "search_space_version": SEARCH_SPACE_VERSION}
    result["fingerprint"] = hashlib.sha256(canonical_json(result).encode()).hexdigest()
    return result


def environment():
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "optuna", "xgboost", "catboost", "joblib", "pyarrow", "scipy", "SQLAlchemy", "threadpoolctl", "matplotlib", "python-decouple"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions, "git_revision": revision}
