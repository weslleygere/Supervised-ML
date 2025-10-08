import os
from typing import List
from datetime import datetime
from src.core.models.factory import RegressionModels 

from dataclasses import dataclass, field

from decouple import config

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
        Timestamped directory created within `output_dir_base` for saving outputs.
    """
    data_path       : str
    schema_path     : str
    output_dir_base : str
    output_dir      : str = field(init=False)

    def __post_init__(self) -> None:
        """Validate paths after initialization."""
        self._validate_file(self.data_path)
        self._validate_file(self.schema_path)
        self._validate_directory(self.output_dir_base)
        self.output_dir = self._create_output_dir(self.output_dir_base)

    @classmethod
    def from_env(cls) -> 'DataConfig':
        """
        Load data configuration from environment variables.

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
    
    @staticmethod
    def _validate_file(file_path: str) -> None:
        """
        Validate that a file exists at the given path.
        
        Parameters
        ----------
        file_path : str
            Path to the file to validate.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    @staticmethod
    def _validate_directory(dir_path: str) -> None:
        """
        Validate that a directory exists or create it.

        Parameters
        ----------
        dir_path : str
            Path to the directory to validate or create.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        elif not os.path.isdir(dir_path):
            raise NotADirectoryError(f"OUTPUT_DIR_base is not a directory: {dir_path}")
        
    @staticmethod
    def _create_output_dir(base_output_dir: str) -> str:
        """
        Create a timestamped output directory.

        Parameters
        ----------
        base_output_dir : str
            Base directory where the output directory will be created.

        Returns
        -------
        output_dir : str
            Path to the newly created output directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, timestamp)
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
    models_config : str
    random_state  : int
    models        : List["RegressionModels"] = field(init=False)

    def __post_init__(self) -> None:
        """Parse `models_config` and populate `self.models`."""
        self.models = self._parse_models(self.models_config)
        self._validate_random_state(self.random_state)

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

    @staticmethod
    def _parse_models(spec: str) -> List["RegressionModels"]:
        """
        Parse the model specification string.

        Parameters
        ----------
        spec : str
            MODELS value from the .env.

        Returns
        -------
        list[RegressionModels]
            List of `RegressionModels` enum members.
        """
        spec = spec.strip()

        if spec.lower() == "all":
            return list(RegressionModels)

        names = [t.strip().upper() for t in spec.split(",") if t.strip()]
        enums = [ModelConfig._to_enum(n) for n in names]
        return ModelConfig._check_duplicates(enums)
    
    @staticmethod
    def _validate_random_state(random_state: int) -> None:
        """
        Validate that random_state is non-negative.
        
        Parameters
        ----------
        random_state : int
            Random seed to validate.
        """
        if random_state < 0:
            raise ValueError(f"RANDOM_STATE must be >= 0, got: {random_state}")


@dataclass
class ValidationConfig:
    """
    Cross-validation configuration settings.
    
    Parameters
    ----------
    n_splits : int
        Number of cross-validation folds.
    """
    n_splits: int

    def __post_init__(self) -> None:
        """Validate n_splits after initialization."""
        self._validate_n_splits(self.n_splits)
    
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
            n_splits=config('N_SPLITS', default=5, cast=int)
        )
    
    @staticmethod
    def _validate_n_splits(n_splits: int) -> None:
        """
        Validate that n_splits is at least 2.
        
        Parameters
        ----------
        n_splits : int
            Number of cross-validation folds to validate.
        """
        if n_splits < 2:
            raise ValueError(f"N_SPLITS must be >= 2, got: {n_splits}") 


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
    log_level : str
    debug     : bool

    def __post_init__(self) -> None:
        """Validate logging settings after initialization."""
        self._validate_log_level(self.log_level)
        self._validate_debug_mode(self.debug)
    
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
    
    @staticmethod
    def _validate_log_level(log_level: str) -> None:
        """
        Validate the logging level.
        
        Parameters
        ----------
        log_level : str
            Logging level to validate.
        """
        if log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid LOG_LEVEL: {log_level}")

        
    @staticmethod
    def _validate_debug_mode(debug: bool) -> None:
        """
        Validate that debug is a boolean.
        
        Parameters
        ----------
        debug : bool
            Debug mode to validate.
        """
        if not isinstance(debug, bool):
            raise ValueError(f"DEBUG must be a boolean, got: {debug}")


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
    data       : DataConfig
    model      : ModelConfig
    validation : ValidationConfig
    logging    : LoggingConfig

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