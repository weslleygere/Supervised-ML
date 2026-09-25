"""Explicit, side-effect-free configuration for the three experiment stages."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import get_type_hints
from decouple import Config, RepositoryEmpty, RepositoryEnv


@dataclass(frozen=True)
class Settings:
    data_path: str = "data/raw/data.parquet"
    schema_path: str = "data/schema/schema.json"
    output_dir: str = "output_experiments"
    run_mode: str = "nested_selection"
    source_run: str = ""
    resume_run: str = ""
    feature_sets: tuple[str, ...] = ("indices", "embeddings", "both")
    aggregation_strategies: tuple[str, ...] = ("mean", "mean_std", "hierarchical")
    reduction_methods: tuple[str, ...] = ("none", "pca")
    pca_component_candidates: tuple[int, ...] = (5, 10, 20, 40, 60)
    pca_variance_candidates: tuple[float, ...] = (0.90, 0.95, 0.99)
    feature_scalings: tuple[str, ...] = ("standard",)
    models: tuple[str, ...] = ("all",)
    selection_efforts: tuple[int, ...] = (1, 5, 10, 15, 25, 50)
    selection_sampling_repeats: int = 10
    outer_splits: int = 5
    outer_repeats: int = 3
    inner_splits: int = 4
    optuna_trials: int = 60
    split_seed: int = 42
    sampling_seed: int = 1042
    search_seed: int = 2042
    model_seed: int = 3042
    model_jobs: int = 1
    cache_entries: int = 8
    recording_start: str = "04:00"
    recording_end: str = "06:00"
    effort_analysis_counts: tuple[int, ...] = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50)
    effort_analysis_repeats: int = 50
    bootstrap_repeats: int = 1000
    confidence_level: float = 0.95
    plateau_tolerance: float = 0.0
    plateau_window: int = 3
    log_level: str = "INFO"

    def __post_init__(self):
        from src.core.models.factory import RegressionModels
        allowed = {
            "run_mode": {"nested_selection", "effort_analysis", "final_fit"},
            "feature_sets": {"indices", "embeddings", "both"},
            "aggregation_strategies": {"mean", "mean_std", "hierarchical"},
            "reduction_methods": {"none", "pca", "pca_variance"},
            "feature_scalings": {"standard", "robust", "none"},
            "models": {"all", *(m.name for m in RegressionModels)},
            "log_level": {"DEBUG", "INFO", "WARNING", "ERROR"},
        }
        for key, choices in allowed.items():
            value = getattr(self, key)
            values = value if isinstance(value, tuple) else (value,)
            if not values or len(values) != len(set(values)) or not set(values) <= choices:
                raise ValueError(f"Invalid {key.upper()}: {value}. Choices: {sorted(choices)}")
        if "all" in self.models and self.models != ("all",):
            raise ValueError("MODELS=all cannot be combined with explicit names.")
        for key in ("selection_efforts", "effort_analysis_counts", "pca_component_candidates"):
            values = getattr(self, key)
            if not values or tuple(sorted(set(values))) != values or min(values) < 1:
                raise ValueError(f"{key.upper()} must be an increasing list of positive integers.")
        if not self.pca_variance_candidates or any(not 0 < v < 1 for v in self.pca_variance_candidates):
            raise ValueError("PCA_VARIANCE_CANDIDATES must contain fractions between zero and one.")
        for key in ("outer_splits", "inner_splits"):
            if getattr(self, key) < 2:
                raise ValueError(f"{key.upper()} must be >= 2.")
        for key in ("outer_repeats", "optuna_trials", "selection_sampling_repeats", "effort_analysis_repeats", "model_jobs", "cache_entries", "plateau_window"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key.upper()} must be positive.")
        for key in ("split_seed", "sampling_seed", "search_seed", "model_seed", "bootstrap_repeats"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key.upper()} cannot be negative.")
        if not 0 < self.confidence_level < 1 or self.plateau_tolerance < 0:
            raise ValueError("Invalid confidence level or plateau tolerance.")
        from datetime import time
        if time.fromisoformat(self.recording_start) >= time.fromisoformat(self.recording_end):
            raise ValueError("The recording window must be increasing within one local day.")
        if self.run_mode != "nested_selection" and not self.source_run:
            raise ValueError("SOURCE_RUN is required for effort_analysis and final_fit.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "Settings":
        hints = get_type_hints(cls)
        return cls(**{k: tuple(v) if getattr(hints[k], "__origin__", None) is tuple else v for k, v in values.items()})

    @classmethod
    def from_env(cls, env_file: str = ".env", overrides: dict | None = None) -> "Settings":
        repository = RepositoryEnv(env_file) if Path(env_file).exists() else RepositoryEmpty()
        valid_keys = {f.name.upper() for f in fields(cls)}
        unknown = set(getattr(repository, "data", {})) - valid_keys
        if unknown:
            raise ValueError(f"Unknown/legacy .env settings: {sorted(unknown)}. See .env.example.")
        config = Config(repository)
        hints = get_type_hints(cls)
        values = {}
        for field in fields(cls):
            raw = config(field.name.upper(), default=None)
            if raw is None:
                continue
            kind = hints[field.name]
            if getattr(kind, "__origin__", None) is tuple:
                subtype = kind.__args__[0]
                values[field.name] = tuple(subtype(x.strip()) for x in raw.split(",") if x.strip())
            else:
                values[field.name] = kind(raw)
        values.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return cls(**values)
