from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Schema:
    """
    Dataset structure used by the instance-MIR experiments.

    Parameters
    ----------
    target : str
        Continuous target column.
    group : str
        Physical location used for grouped cross-validation.
    bag : str
        Capture/deployment identifier used for prediction aggregation.
    instance_columns : tuple[str, ...]
        Columns used to identify individual 1-minute recordings.
    index_prefixes : tuple[str, ...]
        Prefixes identifying acoustic-index columns.
    embedding : str
        Column containing the embedding vector.
    """

    target: str
    group: str
    bag: str
    instance_columns: tuple[str, ...]
    index_prefixes: tuple[str, ...]
    embedding: str

    @classmethod
    def from_dict(cls, schema_dict: dict[str, Any]) -> "Schema":
        """Create a Schema from the parsed JSON configuration."""
        return cls(
            target=schema_dict["target"],
            group=schema_dict["group"],
            bag=schema_dict["bag"],
            instance_columns=tuple(schema_dict["instance_columns"]),
            index_prefixes=tuple(schema_dict["index_prefixes"]),
            embedding=schema_dict["embedding"],
        )

    @property
    def target_names(self) -> list[str]:
        """Return the target column in list form."""
        return [self.target]

    @property
    def metadata_names(self) -> list[str]:
        """Columns required for grouping, aggregation and OOF bookkeeping."""
        return [
            self.group,
            self.bag,
            *self.instance_columns,
        ]

    def index_columns(self, columns: Iterable[str]) -> list[str]:
        """Return acoustic-index columns based on the configured prefixes."""
        return [
            column
            for column in columns
            if column.startswith(self.index_prefixes)
        ]
