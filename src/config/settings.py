import os

from dataclasses import dataclass, field
from datetime import datetime

from decouple import config

from src.core.models.factory import RegressionModels


# =============================================================================
# HELPERS
# =============================================================================


def _parse_csv_strings(
    value: str,
) -> tuple[str, ...]:
    """
    Parse a comma-separated list of lowercase strings.
    """

    values = tuple(
        item.strip().lower()
        for item in str(value).split(",")
        if item.strip()
    )

    if not values:
        raise ValueError(
            "Expected at least one comma-separated value."
        )

    return values


def _parse_csv_ints(
    value: str,
) -> tuple[int, ...]:
    """
    Parse a comma-separated list of integers.
    """

    try:

        values = tuple(
            int(
                item.strip()
            )
            for item in str(
                value
            ).split(",")
            if item.strip()
        )

    except ValueError as exc:

        raise ValueError(
            f"Expected comma-separated integers, got: {value}"
        ) from exc

    if not values:
        raise ValueError(
            "Expected at least one integer value."
        )

    return values


def _check_duplicates(
    values,
    name: str,
):
    """
    Reject duplicated configuration values.
    """

    if len(
        values
    ) != len(
        set(
            values
        )
    ):

        raise ValueError(
            f"{name} contains duplicated values: {values}"
        )

    return values


# =============================================================================
# DATA
# =============================================================================


@dataclass
class DataConfig:
    data_path: str
    schema_path: str
    output_dir_base: str

    output_dir: str = field(
        init=False
    )

    min_capture_days: int = 7
    max_capture_window_days: int = 28

    DATA_EXTENSIONS = frozenset(
        {
            "csv",
            "xlsx",
            "parquet",
            "json",
        }
    )

    def __post_init__(
        self,
    ) -> None:

        self._validate_dataset_file()
        self._validate_schema_file()
        self._validate_output_base()
        self._validate_temporal_curation()

        self.output_dir = (
            self._create_output_dir()
        )

    @classmethod
    def from_env(
        cls,
    ) -> "DataConfig":

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

            min_capture_days=config(
                "MIN_CAPTURE_DAYS",
                default=7,
                cast=int,
            ),

            max_capture_window_days=config(
                "MAX_CAPTURE_WINDOW_DAYS",
                default=28,
                cast=int,
            ),
        )

    def _validate_temporal_curation(
        self,
    ) -> None:

        if self.min_capture_days < 1:
            raise ValueError(
                "MIN_CAPTURE_DAYS must be >= 1."
            )

        if (
            self.max_capture_window_days
            < self.min_capture_days
        ):
            raise ValueError(
                "MAX_CAPTURE_WINDOW_DAYS must be >= "
                "MIN_CAPTURE_DAYS."
            )

    def _validate_dataset_file(
        self,
    ) -> None:

        if not os.path.isfile(
            self.data_path
        ):

            raise FileNotFoundError(
                f"Dataset file not found: "
                f"{self.data_path}"
            )

        extension = (
            self.data_path
            .rsplit(
                ".",
                1,
            )[-1]
            .lower()
        )

        if extension not in self.DATA_EXTENSIONS:

            allowed = ", ".join(
                sorted(
                    self.DATA_EXTENSIONS
                )
            )

            raise ValueError(
                f"Unsupported dataset extension "
                f"'.{extension}'. "
                f"Allowed: {allowed}"
            )

    def _validate_schema_file(
        self,
    ) -> None:

        if not os.path.isfile(
            self.schema_path
        ):

            raise FileNotFoundError(
                f"Schema file not found: "
                f"{self.schema_path}"
            )

        if not self.schema_path.lower().endswith(
            ".json"
        ):

            raise ValueError(
                f"Schema file must be JSON: "
                f"{self.schema_path}"
            )

    def _validate_output_base(
        self,
    ) -> None:

        if not os.path.exists(
            self.output_dir_base
        ):

            os.makedirs(
                self.output_dir_base,
                exist_ok=True,
            )

        elif not os.path.isdir(
            self.output_dir_base
        ):

            raise NotADirectoryError(
                f"OUTPUT_DIR is not a directory: "
                f"{self.output_dir_base}"
            )

    def _create_output_dir(
        self,
    ) -> str:

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


# =============================================================================
# MODELS AND PIPELINE SEARCH SPACE
# =============================================================================


@dataclass
class ModelConfig:
    """
    Define the complete machine-learning pipeline search space.

    A candidate pipeline may differ in:

        model family
        feature representation
        aggregation strategy
        dimensionality reduction
        reduction dimensionality
        model hyperparameters

    Model family is part of the same Optuna search space as the remaining
    pipeline components.
    """

    models_config: str
    random_state: int

    feature_sets_config: str
    aggregation_strategies_config: str
    reduction_methods_config: str

    pca_indices_candidates_config: str
    pca_embeddings_candidates_config: str

    selection_indices_candidates_config: str
    selection_embeddings_candidates_config: str

    models: list[RegressionModels] = field(
        init=False
    )

    feature_sets: tuple[str, ...] = field(
        init=False
    )

    aggregation_strategies: tuple[str, ...] = field(
        init=False
    )

    reduction_methods: tuple[str, ...] = field(
        init=False
    )

    pca_indices_candidates: tuple[int, ...] = field(
        init=False
    )

    pca_embeddings_candidates: tuple[int, ...] = field(
        init=False
    )

    selection_indices_candidates: tuple[int, ...] = field(
        init=False
    )

    selection_embeddings_candidates: tuple[int, ...] = field(
        init=False
    )

    def __post_init__(
        self,
    ) -> None:

        self.models = (
            self._parse_models()
        )

        self.feature_sets = (
            _check_duplicates(
                _parse_csv_strings(
                    self.feature_sets_config
                ),
                "FEATURE_SETS",
            )
        )

        self.aggregation_strategies = (
            _check_duplicates(
                _parse_csv_strings(
                    self.aggregation_strategies_config
                ),
                "AGGREGATION_STRATEGIES",
            )
        )

        self.reduction_methods = (
            _check_duplicates(
                _parse_csv_strings(
                    self.reduction_methods_config
                ),
                "REDUCTION_METHODS",
            )
        )

        self.pca_indices_candidates = (
            _check_duplicates(
                _parse_csv_ints(
                    self.pca_indices_candidates_config
                ),
                "PCA_INDICES_CANDIDATES",
            )
        )

        self.pca_embeddings_candidates = (
            _check_duplicates(
                _parse_csv_ints(
                    self.pca_embeddings_candidates_config
                ),
                "PCA_EMBEDDINGS_CANDIDATES",
            )
        )

        self.selection_indices_candidates = (
            _check_duplicates(
                _parse_csv_ints(
                    self.selection_indices_candidates_config
                ),
                "SELECTION_INDICES_CANDIDATES",
            )
        )

        self.selection_embeddings_candidates = (
            _check_duplicates(
                _parse_csv_ints(
                    self.selection_embeddings_candidates_config
                ),
                "SELECTION_EMBEDDINGS_CANDIDATES",
            )
        )

        self._validate_random_state()
        self._validate_feature_sets()
        self._validate_aggregation_strategies()
        self._validate_reduction_methods()
        self._validate_reduction_candidates()

        from src.core.models.definitions import (
            AbstractModel,
        )

        AbstractModel.seed = (
            self.random_state
        )

    @classmethod
    def from_env(
        cls,
    ) -> "ModelConfig":

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

            feature_sets_config=str(
                config(
                    "FEATURE_SETS",
                    default=(
                        "indices,"
                        "embeddings,"
                        "both"
                    ),
                )
            ),

            aggregation_strategies_config=str(
                config(
                    "AGGREGATION_STRATEGIES",
                    default=(
                        "mean,"
                        "mean_std,"
                        "hierarchical,"
                        "robust_daily,"
                        "dawn_profile,"
                        "dawn_trend"
                    ),
                )
            ),

            reduction_methods_config=str(
                config(
                    "REDUCTION_METHODS",
                    default=(
                        "none,"
                        "pca,"
                        "supervised_selection"
                    ),
                )
            ),

            pca_indices_candidates_config=str(
                config(
                    "PCA_INDICES_CANDIDATES",
                    default=(
                        "2,4,6,8,10,12,"
                        "16,20,24,32,40"
                    ),
                )
            ),

            pca_embeddings_candidates_config=str(
                config(
                    "PCA_EMBEDDINGS_CANDIDATES",
                    default=(
                        "2,4,6,8,10,12,"
                        "16,20,24,32,40"
                    ),
                )
            ),

            selection_indices_candidates_config=str(
                config(
                    "SELECTION_INDICES_CANDIDATES",
                    default=(
                        "2,4,6,8,10,12,"
                        "16,20,24,32,40"
                    ),
                )
            ),

            selection_embeddings_candidates_config=str(
                config(
                    "SELECTION_EMBEDDINGS_CANDIDATES",
                    default=(
                        "2,4,6,8,10,12,"
                        "16,20,24,32,40"
                    ),
                )
            ),
        )

    # =========================================================================
    # MODELS
    # =========================================================================

    def _parse_models(
        self,
    ) -> list[RegressionModels]:

        value = (
            self.models_config.strip()
        )

        if value.lower() == "all":

            return list(
                RegressionModels
            )

        names = [
            name.strip().upper()
            for name in value.split(",")
            if name.strip()
        ]

        if not names:

            raise ValueError(
                "MODELS must contain at least one model."
            )

        models = [
            self._to_enum(
                name
            )
            for name in names
        ]

        return _check_duplicates(
            models,
            "MODELS",
        )

    @staticmethod
    def _to_enum(
        name: str,
    ) -> RegressionModels:

        try:

            return RegressionModels[
                name
            ]

        except KeyError as exc:

            valid = ", ".join(
                model.name
                for model in RegressionModels
            )

            raise ValueError(
                f"Invalid model name '{name}'. "
                f"Valid names: {valid}."
            ) from exc

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_random_state(
        self,
    ) -> None:

        if self.random_state < 0:

            raise ValueError(
                "RANDOM_STATE must be >= 0, "
                f"got: {self.random_state}"
            )

    def _validate_feature_sets(
        self,
    ) -> None:

        valid = {
            "indices",
            "embeddings",
            "both",
        }

        invalid = (
            set(
                self.feature_sets
            )
            - valid
        )

        if invalid:

            raise ValueError(
                "FEATURE_SETS contains invalid values: "
                f"{sorted(invalid)}. "
                f"Valid values are: {sorted(valid)}."
            )

    def _validate_aggregation_strategies(
        self,
    ) -> None:

        valid = {
            "mean",
            "mean_std",
            "hierarchical",
            "robust_daily",
            "dawn_profile",
            "dawn_trend",
        }

        invalid = (
            set(
                self.aggregation_strategies
            )
            - valid
        )

        if invalid:

            raise ValueError(
                "AGGREGATION_STRATEGIES contains invalid values: "
                f"{sorted(invalid)}. "
                f"Valid values are: {sorted(valid)}."
            )

    def _validate_reduction_methods(
        self,
    ) -> None:

        valid = {
            "none",
            "pca",
            "supervised_selection",
        }

        invalid = (
            set(
                self.reduction_methods
            )
            - valid
        )

        if invalid:

            raise ValueError(
                "REDUCTION_METHODS contains invalid values: "
                f"{sorted(invalid)}. "
                f"Valid values are: {sorted(valid)}."
            )

    def _validate_reduction_candidates(
        self,
    ) -> None:

        for name, values in (
            (
                "PCA_INDICES_CANDIDATES",
                self.pca_indices_candidates,
            ),
            (
                "PCA_EMBEDDINGS_CANDIDATES",
                self.pca_embeddings_candidates,
            ),
            (
                "SELECTION_INDICES_CANDIDATES",
                self.selection_indices_candidates,
            ),
            (
                "SELECTION_EMBEDDINGS_CANDIDATES",
                self.selection_embeddings_candidates,
            ),
        ):

            self._validate_positive_integers(
                name=name,
                values=values,
            )

    @staticmethod
    def _validate_positive_integers(
        name: str,
        values: tuple[int, ...],
    ) -> None:

        if any(
            value <= 0
            for value in values
        ):

            raise ValueError(
                f"{name} must contain only "
                "positive integers."
            )

        if tuple(
            sorted(
                values
            )
        ) != values:

            raise ValueError(
                f"{name} must be sorted "
                "in increasing order."
            )


# =============================================================================
# VALIDATION AND MODEL SELECTION
# =============================================================================


@dataclass
class ValidationConfig:
    """
    Holdout and cross-validation configuration.

    Experimental design
    -------------------
    1. The complete dataset is split once by Point into:

           development set
           final test set

    2. The final test set remains completely isolated during pipeline
       selection.

    3. Candidate pipelines are compared on the development set using
       repeated grouped cross-validation.

    4. Every Optuna trial defines one complete pipeline and is evaluated
       using exactly the same cross-validation partitions.

    5. The primary optimization criterion is the mean Point-balanced
       out-of-fold MAE across CV repetitions.

    6. After selection, the winning pipeline is fitted using the complete
       development set and evaluated once on the final test set.
    """

    test_size: float
    hfi_strata: int

    cv_splits: int
    cv_repeats: int

    optuna_trials: int

    def __post_init__(
        self,
    ) -> None:

        self._validate_test_size()
        self._validate_hfi_strata()
        self._validate_cv()
        self._validate_optuna()

    @classmethod
    def from_env(
        cls,
    ) -> "ValidationConfig":

        return cls(
            test_size=config(
                "TEST_SIZE",
                default=0.20,
                cast=float,
            ),

            hfi_strata=config(
                "HFI_STRATA",
                default=4,
                cast=int,
            ),

            cv_splits=config(
                "CV_SPLITS",
                default=5,
                cast=int,
            ),

            cv_repeats=config(
                "CV_REPEATS",
                default=5,
                cast=int,
            ),

            optuna_trials=config(
                "OPTUNA_TRIALS",
                default=400,
                cast=int,
            ),
        )

    # =========================================================================
    # HOLDOUT
    # =========================================================================

    def _validate_test_size(
        self,
    ) -> None:

        if not (
            0.0
            < self.test_size
            < 1.0
        ):

            raise ValueError(
                "TEST_SIZE must be between 0 and 1, "
                f"got: {self.test_size}"
            )

    def _validate_hfi_strata(
        self,
    ) -> None:

        if self.hfi_strata < 2:
            raise ValueError(
                "HFI_STRATA must be >= 2."
            )

    # =========================================================================
    # CROSS-VALIDATION
    # =========================================================================

    def _validate_cv(
        self,
    ) -> None:

        if self.cv_splits < 2:

            raise ValueError(
                "CV_SPLITS must be >= 2."
            )

        if self.cv_repeats < 1:

            raise ValueError(
                "CV_REPEATS must be >= 1."
            )

    # =========================================================================
    # OPTUNA
    # =========================================================================

    def _validate_optuna(
        self,
    ) -> None:

        if self.optuna_trials < 1:

            raise ValueError(
                "OPTUNA_TRIALS must be >= 1."
            )


# =============================================================================
# LOGGING
# =============================================================================


@dataclass
class LoggingConfig:
    log_level: str
    debug: bool

    def __post_init__(
        self,
    ) -> None:

        valid_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if (
            self.log_level.upper()
            not in valid_levels
        ):

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


# =============================================================================
# SETTINGS
# =============================================================================


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
            data=(
                DataConfig.from_env()
            ),

            model=(
                ModelConfig.from_env()
            ),

            validation=(
                ValidationConfig.from_env()
            ),

            logging=(
                LoggingConfig.from_env()
            ),
        )


settings = Settings.from_env()