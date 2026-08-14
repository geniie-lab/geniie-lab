# Standard library
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, List, Optional

# Third-party libraries
from dataclasses_json import DataClassJsonMixin, dataclass_json

@dataclass_json
@dataclass
class QueryExperimentOutput(DataClassJsonMixin):
    session_name: str
    model: str
    task: str
    dataset: str
    topic_id: str
    query: str
    start: int
    size: Optional[int] = 10
    repetition: Optional[str] = 1
    reason: Optional[str] = None
    stage: Optional[str] = "query"
    total_token: Optional[int] = 0
    thinking: Optional[str] = None
    thinking_token: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

@dataclass_json
@dataclass
class RankingExperimentOutput(DataClassJsonMixin):
    session_name: str
    model: str
    ranker: str
    task: str
    dataset: str
    topic_id: str
    doc_ids: List[str]
    start: int
    size: int
    performance: Dict[str, float | int]
    repetition: Optional[str] = 1
    stage: Optional[str] = "ranking"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

@dataclass_json
@dataclass
class ClickExperimentOutput(DataClassJsonMixin):
    session_name: str
    model: str
    task: str
    dataset: str
    topic_id: str
    rankings: List[int]
    repetition: Optional[str] = 1
    reason: Optional[str] = None
    stage: Optional[str] = "click"
    total_token: Optional[int] = 0
    thinking: Optional[str] = None
    thinking_token: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

@dataclass_json
@dataclass
class RelevanceJudgementExperimentOutput(DataClassJsonMixin):
    session_name: str
    model: str
    task: str
    dataset: str
    topic_id: str
    docid: str
    label: str
    qrel_label: Optional[int] = 0
    repetition: Optional[str] = 1
    stage: Optional[str] = "rel_judge"
    total_token: Optional[int] = 0
    thinking: Optional[str] = None
    thinking_token: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

@dataclass_json
@dataclass
class QueryReformulationExperimentOutput(DataClassJsonMixin):
    session_name: str
    model: str
    task: str
    dataset: str
    topic_id: str
    query: str
    start: int
    size: Optional[int] = 10
    repetition: Optional[str] = 1
    reason: Optional[str] = None
    stage: Optional[str] = "reformulation"
    total_token: Optional[int] = 0
    thinking: Optional[str] = None
    thinking_token: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

@dataclass_json
@dataclass
class NextActionOutput(DataClassJsonMixin):
    session_name: str
    model: str
    task: str
    dataset: str
    topic_id: str
    action: str
    action_num: int
    repetition: Optional[str] = 1
    reason: Optional[str] = None
    stage: Optional[str] = "next_action"
    total_token: Optional[int] = 0
    thinking: Optional[str] = None
    thinking_token: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

@dataclass_json
@dataclass
class SubtopicRelevanceJudgementExperimentOutput(DataClassJsonMixin):
    """Record of a per-subtopic relevance judgement (diversity/intent tasks):
    one record per judged document, carrying the label the model assigned to
    every listed subtopic alongside the official per-subtopic qrel labels."""
    session_name: str
    model: str
    task: str
    dataset: str
    topic_id: str
    docid: str
    labels: List[Dict]                           # [{subtopic, label, evidence}, ...]
    # One entry per subtopic shown to the model. Absent from the qrels means
    # nonrelevant (ndeval convention): these collections record nonrelevance
    # per document, not per subtopic (issue #57).
    qrel_labels: Dict[str, int]
    repetition: Optional[str] = 1
    reason: Optional[str] = None
    stage: Optional[str] = "rel_judge"
    total_token: Optional[int] = 0
    thinking: Optional[str] = None
    thinking_token: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
