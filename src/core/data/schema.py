from dataclasses import dataclass
from typing import List, Literal, Dict, Any, ClassVar


@dataclass
class TargetSchema:
    """
    Dataclass representing the schema of the target variable(s).

    Parameters
    ----------
    name : str
        Name of the target column.
    type : Literal['numerical']
        Type of the target variable.
    """
    VALID_TYPES : ClassVar[tuple[str, ...]] = ("numerical",)

    name: str
    type: Literal["numerical"]

    def __post_init__(self) -> None:
        """Validate target fields."""
        self._validate_name()
        self._validate_type()

    def _validate_name(self) -> None:
        """Validate that the target name is not empty."""
        if not self.name.strip():
            raise ValueError("Target name cannot be an empty string.")

    def _validate_type(self) -> None:
        """Validate that the target type is valid."""
        if self.type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid target type '{self.type}' for target '{self.name}'. "
                f"Valid types: {', '.join(self.VALID_TYPES)}"
            )


@dataclass
class FeatureSchema:
    """
    Dataclass representing the schema of a feature column.

    Parameters
    ----------
    name : str
        Name of the feature column.
    type : Literal['numerical', 'ignore']
        Type of the feature variable.
    """
    VALID_TYPES: ClassVar[tuple[str, ...]] = ("numerical", "ignore", "id")

    name: str
    type: Literal["numerical", "ignore", "id"]

    def __post_init__(self) -> None:
        """Validate feature fields."""
        self._validate_name()
        self._validate_type()
    
    def _validate_name(self) -> None:
        """Validate that the feature name is not empty."""
        if not self.name.strip():
            raise ValueError("Feature name cannot be an empty string.")
    
    def _validate_type(self) -> None:
        """Validate that the feature type is valid."""
        if self.type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid feature type '{self.type}' for feature '{self.name}'. "
                f"Valid types: {', '.join(self.VALID_TYPES)}"
            )


@dataclass
class Schema:
    """
    Dataclass representing the full dataset schema, including target and features.

    Parameters
    ----------
    target : TargetSchema
        Schema definition for the target variable(s).
    features : List[FeatureSchema]
        List of schema definitions for each feature column.
    """
    target  : List[TargetSchema]
    features: List[FeatureSchema]

    def __post_init__(self) -> None:
        """Validate schema integrity."""
        self._validate_no_duplicate_names()
        self._validate_no_target_feature_overlap()

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> "Schema":
        """
        Create a Schema instance from a dictionary (parsed from JSON schema file).

        Parameters
        ----------
        schema_dict : Dict[str, Any]
            Dictionary representation of the schema.

        Returns
        -------
        Schema
            Instantiated Schema object.
        """
        try:
            target = [
                TargetSchema(
                    name=entry["name"],
                    type=entry["type"]
                )
                for entry in schema_dict["targets"]
            ]
            
            features = [
                FeatureSchema(
                    name=entry["name"],
                    type=entry["type"]
                )
                for entry in schema_dict["features"]
            ]
            
            return cls(target=target, features=features)
            
        except KeyError as e:
            raise ValueError(f"Missing key in schema entry: {e}") from e

        except Exception as e:
            raise ValueError(f"Invalid schema structure: {e}") from e
        
    @property
    def target_names(self) -> List[str]:
        """
        Get all target names.
        
        Returns
        -------
        List[str]
            List of target names.
        """
        return [t.name for t in self.target]
    
    @property
    def feature_names(self) -> List[str]:
        """
        Get all feature names.
        
        Returns
        -------
        List[str]
            List of feature names.
        """
        return [f.name for f in self.features]
    
    @property
    def valid_feature_names(self) -> List[str]:
        """
        Get all feature names that are not 'ignore' or 'id'.
        
        Returns
        -------
        List[str]
            List of valid feature names.
        """
        return [f.name for f in self.features if f.type not in ("ignore", "id")]
    
    @property
    def valid_feature_names_id(self) -> List[str]:
        """
        Get all feature names that are not 'ignore'.
        
        Returns
        -------
        List[str]
            List of valid feature names.
        """
        return [f.name for f in self.features if f.type not in "ignore"]
    
    @property
    def ignored_feature_names(self) -> List[str]:
        """
        Get all feature names that are 'ignore'.
        
        Returns
        -------
        List[str]
            List of ignored feature names.
        """
        return [f.name for f in self.features if f.type == "ignore"]
        
    def _validate_no_duplicate_names(self) -> None:
        """Ensure there are no duplicate names among targets and features."""
        all_names = [t.name for t in self.target] + [f.name for f in self.features]
        duplicates = {name for name in all_names if all_names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate names found in schema: {', '.join(duplicates)}")
        
    def _validate_no_target_feature_overlap(self) -> None:
        """Ensure that target names do not overlap with feature names."""
        target_names = {t.name for t in self.target}
        feature_names = {f.name for f in self.features}
        overlap = target_names.intersection(feature_names)
        if overlap:
            raise ValueError(
                f"Target and feature names overlap: {', '.join(overlap)}"
            )
