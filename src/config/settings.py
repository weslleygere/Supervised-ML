import os
from dataclasses import dataclass, field
from datetime import datetime

from decouple import config

from src.core.models.factory import RegressionModels


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None

    value = str(value).strip()

    if not value or value.lower() == "none":
        return None

    return int(value)


@dataclass
class DataConfig:
    data_path: str
    schema_path: str
    output_dir_base: str
    output_dir: str = field(init=False)

    DATA_EXTENSIONS = frozenset(
        {"csv", "xlsx", "parquet", "json"}
    )

    def __post_init__(self) -> None:
        self._validate_dataset_file()
        self._validate_schema_file()
        self._validate_output_base()
        self.output_dir = self._create_output_dir()

    @classmethod
    def from_env(cls) -> "DataConfig":
        return cls(
            data_path=str(
                config(
                    "DATA_PATH",
                    default="data/raw/data.csv",
                )
            ),
            schema_path=str(
                config(
                    "SCHEMA_PATH",
                    default="data/schema/schema.json",
                )
            ),
            output_dir_base=str(
                config(
                    "OUTPUT_DIR",
                    default="output",
                )
            ),
        )

    def _validate_dataset_file(self) -> None:
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_path}"
            )

        extension = (
            self.data_path
            .rsplit(".", 1)[-1]
            .lower()
        )

        if extension not in self.DATA_EXTENSIONS:
            allowed = ", ".join(
                sorted(self.DATA_EXTENSIONS)
            )

            raise ValueError(
                f"Unsupported dataset extension '.{extension}'. "
                f"Allowed: {allowed}"
            )

    def _validate_schema_file(self) -> None:
        if not os.path.isfile(self.schema_path):
            raise FileNotFoundError(
                f"Schema file not found: {self.schema_path}"
            )

        if not self.schema_path.lower().endswith(".json"):
            raise ValueError(
                f"Schema file must be a JSON file: "
                f"{self.schema_path}"
            )

    def _validate_output_base(self) -> None:
        if not os.path.exists(self.output_dir_base):
            os.makedirs(
                self.output_dir_base,
                exist_ok=True,
            )

        elif not os.path.isdir(self.output_dir_base):
            raise NotADirectoryError(
                f"OUTPUT_DIR is not a directory: "
                f"{self.output_dir_base}"
            )

    def _create_output_dir(self) -> str:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_dir = os.path.join(
            self.output_dir_base,
            timestamp,
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        return output_dir


@dataclass
class ModelConfig:
    models_config: str
    random_state: int
    feature_set: str
    pca_components: int | None

    models: list[RegressionModels] = field(
        init=False
    )

    def __post_init__(self) -> None:
        self.models = self._parse_models()
        self._validate_random_state()
        self._validate_feature_set()
        self._validate_pca_components()

        from src.core.models.definitions import (
            AbstractModel,
        )

        AbstractModel.seed = self.random_state

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            models_config=str(
                config(
                    "MODELS",
                    default="all",
                )
            ),
            random_state=config(
                "RANDOM_STATE",
                default=42,
                cast=int,
            ),
            feature_set=str(
                config(
                    "FEATURE_SET",
                    default="indices",
                )
            ).lower(),
            pca_components=_optional_int(
                str(
                    config(
                        "PCA_COMPONENTS",
                        default="50",
                    )
                )
            ),
        )

    def _parse_models(
        self,
    ) -> list[RegressionModels]:

        value = self.models_config.strip()

        if value.lower() == "all":
            return list(RegressionModels)

        names = [
            name.strip().upper()
            for name in value.split(",")
            if name.strip()
        ]

        models = [
            self._to_enum(name)
            for name in names
        ]

        return self._check_duplicates(models)

    def _validate_random_state(self) -> None:
        if self.random_state < 0:
            raise ValueError(
                "RANDOM_STATE must be >= 0, "
                f"got: {self.random_state}"
            )

    def _validate_feature_set(self) -> None:
        valid = {
            "indices",
            "embeddings",
            "both",
        }

        if self.feature_set not in valid:
            raise ValueError(
                "FEATURE_SET must be one of: "
                "indices, embeddings, both."
            )

    def _validate_pca_components(self) -> None:
        if (
            self.pca_components is not None
            and self.pca_components <= 0
        ):
            raise ValueError(
                "PCA_COMPONENTS must be > 0 or None."
            )

    @staticmethod
    def _to_enum(
        name: str,
    ) -> RegressionModels:

        try:
            return RegressionModels[name]

        except KeyError as exc:
            valid = ", ".join(
                model.name
                for model in RegressionModels
            )

            raise ValueError(
                f"Invalid model name '{name}'. "
                f"Valid names: {valid}."
            ) from exc

    @staticmethod
    def _check_duplicates(
        models: list[RegressionModels],
    ) -> list[RegressionModels]:

        seen = set()

        for model in models:
            if model in seen:
                raise ValueError(
                    f"Duplicate model "
                    f"'{model.name}' found in MODELS."
                )

            seen.add(model)

        return models


@dataclass
class ValidationConfig:
    outer_splits: int
    inner_splits: int
    optuna_trials: int

    def __post_init__(self) -> None:
        if self.outer_splits < 2:
            raise ValueError(
                "OUTER_SPLITS must be >= 2."
            )

        if self.inner_splits < 2:
            raise ValueError(
                "INNER_SPLITS must be >= 2."
            )

        if self.optuna_trials < 1:
            raise ValueError(
                "OPTUNA_TRIALS must be >= 1."
            )

    @classmethod
    def from_env(
        cls,
    ) -> "ValidationConfig":

        return cls(
            outer_splits=config(
                "OUTER_SPLITS",
                default=5,
                cast=int,
            ),
            inner_splits=config(
                "INNER_SPLITS",
                default=4,
                cast=int,
            ),
            optuna_trials=config(
                "OPTUNA_TRIALS",
                default=20,
                cast=int,
            ),
        )


@dataclass
class LoggingConfig:
    log_level: str
    debug: bool

    def __post_init__(self) -> None:
        if self.log_level.upper() not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }:
            raise ValueError(
                f"Invalid LOG_LEVEL: "
                f"{self.log_level}"
            )

    @classmethod
    def from_env(
        cls,
    ) -> "LoggingConfig":

        return cls(
            log_level=str(
                config(
                    "LOG_LEVEL",
                    default="INFO",
                )
            ),
            debug=config(
                "DEBUG",
                default=False,
                cast=bool,
            ),
        )


@dataclass
class Settings:
    data: DataConfig
    model: ModelConfig
    validation: ValidationConfig
    logging: LoggingConfig

    @classmethod
    def from_env(
        cls,
    ) -> "Settings":

        return cls(
            data=DataConfig.from_env(),
            model=ModelConfig.from_env(),
            validation=ValidationConfig.from_env(),
            logging=LoggingConfig.from_env(),
        )


settings = Settings.from_env()
