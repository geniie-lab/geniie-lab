from dataclasses import dataclass, field
from typing import List, Optional, Iterator, TypeVar, Generic, Type
import random

# Base topic
@dataclass
class BaseTopic:
    id: str
    title: str

    @classmethod
    def from_ir_datasets(cls, raw) -> "BaseTopic":
        return cls(id=raw.query_id, title=raw.title)

    def __str__(self):
        return f"- **Title**: {self.title}"

@dataclass
class TitleOnlyTopic(BaseTopic):
    @classmethod
    def from_ir_datasets(cls, raw):
        title = getattr(raw, "text", getattr(raw, "title", ""))
        return cls(id=raw.query_id, title=title)

@dataclass
class TitleDescriptionTopic(BaseTopic):
    description: str

    @classmethod
    def from_ir_datasets(cls, raw) -> "TitleDescriptionTopic":
        return cls(id=raw.query_id, title=raw.title, description=raw.description)

    def __str__(self):
        desc = self.description.replace("\n", " ")
        return f"- **Title**: {self.title}\n- **Description**: {desc}"

@dataclass
class TitleNarrativeTopic(BaseTopic):
    narrative: str

    @classmethod
    def from_ir_datasets(cls, raw) -> "TitleNarrativeTopic":
        return cls(id=raw.query_id, title=raw.title, narrative=raw.narrative)

    def __str__(self):
        narr = self.narrative.replace("\n", " ")
        return f"- **Title**: {self.title}\n- **Narrative**: {narr}"

@dataclass
class TitleDescriptionNarrativeTopic(BaseTopic):
    description: Optional[str] = None
    narrative: Optional[str] = None
    reldoc: Optional[int] = None

    @classmethod
    def from_ir_datasets(cls, raw) -> "TitleDescriptionNarrativeTopic":
        return cls(
            id=raw.query_id,
            title=raw.title,
            description=raw.description,
            narrative=raw.narrative,
        )

    def __str__(self):
        parts = [f"- **Title**: {self.title}"]
        if self.description:
            parts.append(f"- **Description**: {self.description.replace('\n', ' ')}")
        if self.narrative:
            parts.append(f"- **Narrative**: {self.narrative.replace('\n', ' ')}")
        return "\n".join(parts)

@dataclass
class FullTopic(TitleDescriptionNarrativeTopic):
    pass

T = TypeVar("T", bound=BaseTopic)

@dataclass
class TopicList(Generic[T]):
    _topics: List[T] = field(default_factory=list)

    def append(self, topic: T):
        self._topics.append(topic)

    def __getitem__(self, index: int) -> T:
        return self._topics[index]

    def __len__(self) -> int:
        return len(self._topics)

    def __iter__(self) -> Iterator[T]:
        return iter(self._topics)

    def random(self) -> Optional[T]:
        return random.choice(self._topics) if self._topics else None
