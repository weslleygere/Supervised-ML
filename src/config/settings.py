import os
from typing import List
from datetime import datetime
from dataclasses import dataclass, field

from decouple import config

from src.core.models.factory import RegressionModels 


@dataclass
class DataConfig:
    """
    Data-related configuration settings.
    
    Parameters
    ----------
    data_path : str
        Path to the input dataset.
    schema_path : str
        Path to the input schema file.
    output_dir_base : str
        Directory where output files will be saved.

    Attributes
    ----------
    output_dir : str
        Timestamped directory created within `output_dir_base` to store results.
    """
    DATA_EXTENSIONS = frozenset({"csv", "xlsx", "parquet", "json"})

    data_path      : str
    schema_path    : str
    output_dir_base: str
    output_dir     : str = field(init=False)

    def __post_init__(self) -> None:
        """Validate input paths and initialize the timestamped output directory."""
        self._validate_dataset_file()
        self._validate_schema_file()
        self._validate_output_base()
        self.output_dir = self._create_output_dir()

    @classmethod
    def from_env(cls) -> "DataConfig":
        """
        Load configuration settings from environment variables.

        Returns
        -------
        DataConfig
            An instance of DataConfig populated from environment variables.
        """
        return cls(
            data_path       = str(config('DATA_PATH', default='data/raw/data.csv')),
            schema_path     = str(config('SCHEMA_PATH', default='data/schema/schema.json')),
            output_dir_base = str(config('OUTPUT_DIR', default='output'))
        )

    def _validate_dataset_file(self) -> None:
        """Validate that the dataset file exists and has a supported extension."""
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        ext = self.data_path.split(".")[-1].lower()
        if ext not in DataConfig.DATA_EXTENSIONS:
            allowed = ", ".join(sorted(DataConfig.DATA_EXTENSIONS))
            raise ValueError(
                f"Unsupported dataset extension '.{ext}'. Allowed: {allowed}"
            )

    def _validate_schema_file(self) -> None:
        """Validate that the schema file exists and is a JSON file."""
        if not os.path.isfile(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        if not self.schema_path.lower().endswith(".json"):
            raise ValueError(f"Schema file must be a JSON file: {self.schema_path}")
        
    def _validate_output_base(self) -> None:
        """Validate or create the base output directory."""
        if not os.path.exists(self.output_dir_base):
            os.makedirs(self.output_dir_base, exist_ok=True)
        elif not os.path.isdir(self.output_dir_base):
            raise NotADirectoryError(f"OUTPUT_DIR is not a directory: {self.output_dir_base}")

    def _create_output_dir(self) -> str:
        """
        Create a timestamped output directory within the base output directory.

        Returns
        -------
        str
            Full path to the created timestamped output directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir_base, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir


@dataclass
class ModelConfig:
    """
    Model-related configuration settings.

    Parameters
    ----------
    models_config : str
        Raw string from .env describing which models to include.
        Accepted formats:
            - 'all'
            - 'M1,M2,M3'
    random_state : int
        Random seed for model training.

    Attributes
    ----------
    models : list[RegressionModels]
        Parsed list of models (enum members) to include in the evaluation.
    """
    models_config: str
    random_state : int
    models       : List["RegressionModels"] = field(init=False)

    def __post_init__(self) -> None:
        """Parse `models_config` and populate `self.models`."""
        self.models = self._parse_models()
        self._validate_random_state()

        from src.core.models.definitions import AbstractModel
        AbstractModel.seed = self.random_state

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """
        Load configuration from environment variables.

        Returns
        -------
        ModelConfig
            An instance of ModelConfig populated from environment variables.
        """
        return cls(
            models_config = str(config("MODELS", default="all")),
            random_state  = config("RANDOM_STATE", default=42, cast=int),
        )
    
    def _parse_models(self) -> List["RegressionModels"]:
        """
        Parse the model specification string.

        Returns
        -------
        list[RegressionModels]
            List of `RegressionModels` enum members.
        """
        models_config = self.models_config.strip()

        if models_config.lower() == "all":
            return list(RegressionModels)

        names = [t.strip().upper() for t in models_config.split(",") if t.strip()]
        enums = [ModelConfig._to_enum(n) for n in names]
        return ModelConfig._check_duplicates(enums)
    
    def _validate_random_state(self) -> None:
        """Validate that random_state is non-negative."""
        if self.random_state < 0:
            raise ValueError(f"RANDOM_STATE must be >= 0, got: {self.random_state}")

    @staticmethod
    def _to_enum(name: str) -> "RegressionModels":
        """
        Convert a token to a `RegressionModels` enum member.

        Parameters
        ----------
        name : str
            Model name token to convert.

        Returns
        -------
        RegressionModels
            Matching enum member.
        """
        try:
            return RegressionModels[name]
        except KeyError as e:
            valid = ", ".join(m.name for m in RegressionModels)
            raise ValueError(
                f"Invalid model name '{name}'. Valid names: {valid}."
            ) from e

    @staticmethod
    def _check_duplicates(items: List["RegressionModels"]) -> List["RegressionModels"]:
        """
        Ensure there are no duplicates in the model list.

        Parameters
        ----------
        items : list[RegressionModels]
            List of models to validate.

        Returns
        -------
        list[RegressionModels]
            The same list if no duplicates are present.
        """
        seen = set()
        for x in items:
            if x in seen:
                raise ValueError(
                    f"Duplicate model '{x.name}' found in MODELS specification. "
                    "Remove duplicates and try again."
                )
            seen.add(x)
        return items


@dataclass
class ValidationConfig:
    """
    Cross-validation configuration settings.
    
    Parameters
    ----------
    n_splits : int
        Number of cross-validation folds.
    cv_strategy : str
        Cross-validation strategy ('kfold' or 'groupkfold').
    group_column : str
        Column name used for GroupKFold.
    """
    n_splits    : int
    cv_strategy : str
    group_column: str

    def __post_init__(self) -> None:
        """Validate cross-validation settings after initialization."""
        self._validate_n_splits()
        self._validate_cv_strategy()
        self._validate_group_column()
    
    @classmethod
    def from_env(cls) -> 'ValidationConfig':
        """
        Load validation configuration from environment variables.
        
        Returns
        -------
        ValidationConfig
            An instance of ValidationConfig populated from environment variables.
        """
        return cls(
            n_splits=config('N_SPLITS', default=5, cast=int),
            cv_strategy=str(config('CV_STRATEGY', default='kfold')).lower(),
            group_column=str(config('GROUP_COLUMN', default='')).strip()
        )
    
    def _validate_n_splits(self) -> None:
        """Validate that n_splits is at least 2."""
        if self.n_splits < 2:
            raise ValueError(f"N_SPLITS must be >= 2, got: {self.n_splits}")
    
    def _validate_cv_strategy(self) -> None:
        """Validate that cv_strategy is a supported strategy."""
        valid_strategies = {"kfold", "groupkfold"}
        if self.cv_strategy not in valid_strategies:
            raise ValueError(f"CV_STRATEGY must be one of {valid_strategies}, got: {self.cv_strategy}")
    
    def _validate_group_column(self) -> None:
        """Validate group_column based on cv_strategy."""
        if self.cv_strategy == "groupkfold" and self.group_column == "":
            raise ValueError(
                "GROUP_COLUMN must be set when CV_STRATEGY='groupkfold'"
            )


@dataclass
class LoggingConfig:
    """
    Logging configuration settings.

    Parameters
    ----------
    log_level : str
        Logging level (e.g., 'DEBUG', 'INFO').
    debug : bool
        Enable debug mode.
    """
    log_level: str
    debug    : bool

    def __post_init__(self) -> None:
        """Validate logging settings after initialization."""
        self._validate_log_level()
        self._validate_debug_mode()
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """
        Load logging configuration from environment variables.

        Returns
        -------
        LoggingConfig
             An instance of LoggingConfig populated from environment variables.
        """
        return cls(
            log_level = str(config('LOG_LEVEL', default='INFO')),
            debug     = config('DEBUG', default=False, cast=bool)
        )
    
    def _validate_log_level(self) -> None:
        """Validate the logging level."""
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid LOG_LEVEL: {self.log_level}")

        
    def _validate_debug_mode(self) -> None:
        """Validate that debug is a boolean."""
        if not isinstance(self.debug, bool):
            raise ValueError(f"DEBUG must be a boolean, got: {self.debug}")


@dataclass
class Settings:
    """
    Settings container aggregating all configuration sections.

    Parameters
    ----------
    data : DataConfig
        Data-related settings.
    model : ModelConfig
        Model-related settings.
    validation : ValidationConfig
        Validation-related settings.
    logging : LoggingConfig
        Logging-related settings.
    """
    data      : DataConfig
    model     : ModelConfig
    validation: ValidationConfig
    logging   : LoggingConfig

    @classmethod
    def from_env(cls) -> 'Settings':
        """
        Load all settings from environment variables.
        
        Returns
        -------
        Settings
            An instance of Settings populated from environment variables.
        """
        return cls(
            data       = DataConfig.from_env(),
            model      = ModelConfig.from_env(),
            validation = ValidationConfig.from_env(),
            logging    = LoggingConfig.from_env()
        )


settings = Settings.from_env()
