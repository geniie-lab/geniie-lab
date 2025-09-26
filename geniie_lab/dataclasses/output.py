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
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
