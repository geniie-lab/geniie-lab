from dataclasses import dataclass
from typing import List, Optional, Type
from ir_measures.measures.base import Measure
from geniie_lab.dataclasses.topic import BaseTopic, FullTopic

@dataclass
class TaskDescription:
    name: str
    description: str
    measurement: List[Measure]
    start_offset: int = 0
    serp_size: int = 10
    termination_condition: Optional[str] = None
    
@dataclass
class TopicDescription:
    name: str
    type: str
    topic_class: Optional[Type[BaseTopic]] = FullTopic

@dataclass
class CorpusDescription:
    name: str
    description: str
    index_name: str

@dataclass
class ModelDescription:
    type: str
    name: str
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    system_prompt: Optional[str] = "You're a helpful assistant"
    system_role: Optional[str] = None

@dataclass
class ToolDescription:
    name: str
    ranking_model: str
    index_name: str
    description: str
    host: str = "localhost"
    port: int = 9200
    encode_model: Optional[str] = None