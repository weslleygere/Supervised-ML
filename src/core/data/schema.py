from typing import List, Literal, Dict, Any, ClassVar

from dataclasses import dataclass

@dataclass
class TargetSchema:
    """
    Dataclass representing the target variable(s) schema.

    Parameters
    ----------
    names : List[str]
        List of column names that are considered target variables.
    """
    names: List[str]

    def __post_init__(self) -> None:
        """Validate that target names is a non-empty list."""
        if not isinstance(self.names, list) or not self.names:
            raise ValueError("Target 'names' must be a non-empty list.")


@dataclass
class FeatureSchema:
    """
    Dataclass representing the schema of a single feature.

    Parameters
    ----------
    name : str
        Name of the feature column.
    type : Literal['numerical', 'ignore']
        Type of the feature. Must be either 'numerical' or 'ignore'.
    """
    VALID_TYPES: ClassVar[tuple[str, ...]] = ("numerical", "ignore")

    name : str
    type : Literal["numerical", "ignore"]

    def __post_init__(self) -> None:
        """Validate that feature type is one of the accepted values."""
        if self.type not in self.VALID_TYPES:
            valid = ", ".join(self.VALID_TYPES)
            raise ValueError(
                f"Invalid feature type '{self.type}' for feature '{self.name}'. "
                f"Valid types: {valid}"
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
    target   : TargetSchema
    features : List[FeatureSchema]

    def __post_init__(self) -> None:
        """Validate the schema for consistency and completeness."""
        if not self.features:
            raise ValueError("At least one feature must be provided")
        
        feature_names = [f.name for f in self.features]
        all_names = feature_names + self.target.names
        
        unique_names = set(all_names)
        if len(all_names) != len(unique_names):
            duplicates = [name for name in unique_names if all_names.count(name) > 1]
            raise ValueError(f"Duplicate names found: {', '.join(sorted(duplicates))}")
        
        target_set = set(self.target.names)
        feature_set = set(feature_names)
        overlap = target_set & feature_set
        
        if overlap:
            raise ValueError(f"Target columns cannot be features: {', '.join(sorted(overlap))}")

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> "Schema":
        """
        Create a Schema instance from a dictionary.

        Parameters
        ----------
        schema_dict : Dict[str, Any]
            Dictionary containing the schema definition with "target" and "features" keys.

        Returns
        -------
        schema : Schema
            Instantiated Schema object containing target and feature definitions based on the provided dictionary.
        """
        try:
            target = TargetSchema(names=schema_dict["target"]["names"])
            
            features = [
                FeatureSchema(name=entry["name"], type=entry["type"])
                for entry in schema_dict["features"]
            ]
            
            return cls(target=target, features=features)
            
        except KeyError as e:
            raise ValueError(f"Missing required key in schema") from e
        except Exception as e:
            raise ValueError(f"Invalid schema structure") from e