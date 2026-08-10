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

# Subtopic-bearing topics for diversity/intent tasks (issue #51).
#
# Subtopics are the evaluation ground truth of diversity collections; they are
# NEVER rendered into prompts. __str__ (what stages inject into the LLM
# context) shows only the searcher-visible fields, with no disclosure option.

@dataclass
class Subtopic:
    number: str
    text: str
    probability: Optional[float] = None   # NTCIR intent probability
    intent_type: Optional[str] = None     # NTCIR-10 'nav' / 'inf'

@dataclass
class SubtopicTopic(BaseTopic):
    """Base for topics that carry subtopics. Subtopics are held for logging
    and evaluation only; rendering exposes just the searcher-visible fields."""
    subtopics: List[Subtopic] = field(default_factory=list)

    def __str__(self):
        return f"- **Title**: {self.title}"

@dataclass
class TrecDiversityTopic(SubtopicTopic):
    """TREC Web Track diversity topic (2009-2012): query text, description,
    topic type (faceted/ambiguous), and the judged subtopics.

    Rendering is query-only, matching the track protocol: participants were
    given only the query field; the description (assessor-facing) and the
    subtopics were released after run submission (TREC 2009 Web Track
    overview, section 2). Both are kept as data for logging and analysis."""
    description: Optional[str] = None
    topic_type: Optional[str] = None

    @classmethod
    def from_ir_datasets(cls, raw) -> "TrecDiversityTopic":
        description = getattr(raw, "description", None)
        return cls(
            id=raw.query_id,
            title=" ".join(raw.query.split()),
            description=" ".join(description.split()) if description else None,
            topic_type=getattr(raw, "type", None),
            subtopics=[
                Subtopic(number=str(getattr(s, "number", i + 1)),
                         text=" ".join(s.text.split()),
                         intent_type=getattr(s, "type", None) or None)
                for i, s in enumerate(raw.subtopics)
            ],
        )

@dataclass
class NtcirIntentTopic(SubtopicTopic):
    """NTCIR INTENT-1/2 Japanese Document Ranking topic: the searcher sees the
    query string alone; intents carry probabilities and (INTENT-2) nav/inf
    types for evaluation."""

    @classmethod
    def from_ir_datasets(cls, raw) -> "NtcirIntentTopic":
        return cls(
            id=raw.query_id,
            title=raw.query,
            subtopics=[
                Subtopic(
                    number=str(s.number),
                    text=getattr(s, "description", ""),
                    probability=getattr(s, "probability", None),
                    intent_type=getattr(s, "intent_type", None) or None,
                )
                for s in raw.subtopics
            ],
        )

# Subtopic-presenting variants: same data as their parents, but __str__ (the
# text injected into prompts) includes the subtopic components. Which class
# an experiment uses is its presentation choice, exactly like the
# TitleOnly/TitleDescription family above. Selection policies (e.g. showing
# only the most probable intents) belong to the experiment, which can
# subclass and filter; these classes present everything they carry.

@dataclass
class TrecDiversitySubtopicsTopic(TrecDiversityTopic):
    """TREC diversity topic presenting all available information: title,
    description, topic type, and the numbered subtopics with their types."""

    def __str__(self):
        parts = [f"- **Title**: {self.title}"]
        if self.description:
            parts.append(f"- **Description**: {self.description}")
        if self.topic_type:
            parts.append(f"- **Topic type**: {self.topic_type}")
        if self.subtopics:
            parts.append("- **Subtopics**:")
            for s in self.subtopics:
                tag = f" ({s.intent_type})" if s.intent_type else ""
                parts.append(f"  {s.number}.{tag} {s.text}")
        return "\n".join(parts)

@dataclass
class NtcirIntentSubtopicsTopic(NtcirIntentTopic):
    """NTCIR INTENT topic presenting all available information: query and the
    numbered intents with probabilities and (where present) nav/inf types."""

    def __str__(self):
        parts = [f"- **Title**: {self.title}"]
        if self.subtopics:
            parts.append("- **Subtopics**:")
            for s in self.subtopics:
                attrs = []
                if s.probability is not None:
                    attrs.append(f"p={s.probability:.3f}")
                if s.intent_type:
                    attrs.append(s.intent_type)
                tag = f" ({', '.join(attrs)})" if attrs else ""
                parts.append(f"  {s.number}.{tag} {s.text}")
        return "\n".join(parts)

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
