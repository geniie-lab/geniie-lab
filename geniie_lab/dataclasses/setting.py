# Standard library
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Union

# Local application imports
from geniie_lab.dataclasses.serp import Serp, FullText
from geniie_lab.response import Clicks, Query, RelevanceJudgement, SubtopicRelevanceJudgement, Action
from geniie_lab.dataclasses.description import (
    CorpusDescription,
    ModelDescription,
    TaskDescription,
    ToolDescription,
    TopicDescription,
)
from geniie_lab.dataclasses.topic import (
    TitleDescriptionNarrativeTopic, FullTopic,
    TitleDescriptionTopic,
    TitleNarrativeTopic,
    TitleOnlyTopic
)
from geniie_lab.memory import ConversationHistory

@dataclass
class StageConfig:
    instruction: Optional[str] = None
    # Opt-in: when True, the stage's record includes the model's reasoning
    # trace text (thinking); when False (default) the text is logged as null.
    # thinking_token (the trace's estimated token count) is always recorded,
    # so per-stage thinking cost stays measurable. Generation is unaffected —
    # this only controls what lands in the session log.
    log_thinking: bool = False
    # Optional structured-output schema override for the stage. Currently
    # honoured by the relevance stage: set to SubtopicRelevanceJudgement[scale]
    # for per-subtopic labelling on diversity/intent topics (the document is
    # judged against every subtopic in one call). The scale must match the
    # labels the stage instruction defines.
    response_model: Optional[type] = None

@dataclass
class ExperimentSettings:
    name: str
    task: TaskDescription
    topicset: TopicDescription
    corpus: CorpusDescription
    models: List[ModelDescription]
    tools: List[ToolDescription]
    stages: Dict[Literal["query", "click", "relevance", "reformulate"], StageConfig] = field(default_factory=dict)
    loop_num_per_topic: int = 1
    plan: Optional[List[str]] = None  # List of stage names to execute in order
    max_topics: Optional[int] = None # None means all topics
    topic_ids: Optional[str] = None # e.g. "1:10"
    max_actions: Optional[int] = None
    custom_settings: Optional[str] = None
    full_log: Optional[bool] = False
    # Set False to reproduce runs made before visited marking existed.
    mark_visited_results: bool = True

@dataclass
class ExperimentState:
    topic: Union[TitleOnlyTopic, TitleDescriptionTopic, TitleNarrativeTopic, TitleDescriptionNarrativeTopic, FullTopic]
    memory: ConversationHistory
    query: Optional[Query] = None
    serp: Optional[Serp] = None
    docids: Optional[List[str]] = None
    clicks: Optional[Clicks] = None
    fulltext: Optional[FullText] = None
    relevance_judgement: Optional[Union[RelevanceJudgement, SubtopicRelevanceJudgement]] = None
    error: Optional[str] = None
    action_num: Optional[int] = 1
    next_action: Optional[Action] = None
    visited_docids: Set[str] = field(default_factory=set)

@dataclass
class Error:
    error_text: str