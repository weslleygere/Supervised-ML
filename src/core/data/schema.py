from dataclasses import dataclass


@dataclass(frozen=True)
class Schema:
    target: str
    group: str
    bag: str
    audio: str
    datetime: str
    index_prefixes: tuple[str, ...]
    embedding: str

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "Schema":
        return cls(
            target=data["target"],
            group=data["group"],
            bag=data["bag"],
            audio=data["audio"],
            datetime=data["datetime"],
            index_prefixes=tuple(
                data["index_prefixes"]
            ),
            embedding=data["embedding"],
        )

    @property
    def target_names(self) -> list[str]:
        return [
            self.target,
        ]

    @property
    def metadata_names(self) -> list[str]:
        return [
            self.group,
            self.bag,
            self.audio,
            self.datetime,
        ]

    def index_columns(
        self,
        columns,
    ) -> list[str]:
        return [
            col
            for col in columns
            if col.startswith(
                self.index_prefixes
            )
        ]
