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
    token_length: int
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    system_prompt: Optional[str] = "You're a helpful assistant"
    system_role: Optional[str] = None
    # Cap on reasoning tokens for reasoning models, enforced server-side by
    # vLLM (qwen3/deepseek_r1 parsers; pair with a reasoning_end_str transition
    # phrase in the server's --reasoning-config for a graceful cut-off).
    # None = uncapped; ignored by providers without the feature.
    thinking_token_budget: Optional[int] = None
    # Reasoning-effort dial for models trained with one (gpt-oss:
    # "low" | "medium" | "high"; the model plans its own reasoning length).
    # None = provider default; ignored by models without the knob.
    reasoning_effort: Optional[str] = None
    # Send the response schema in the prompt as well as in response_format.
    # None = the provider's own setting; providers with no grammar always do.
    schema_in_prompt: Optional[bool] = None

@dataclass
class ToolDescription:
    name: str
    ranking_model: str
    index_name: str
    description: str
    host: str = "localhost"
    port: int = 9200
    use_ssl: bool = True
    encode_model: Optional[str] = None