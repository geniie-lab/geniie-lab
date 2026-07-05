# Standard library
from typing import Callable, Protocol, Tuple

# Local application imports
from geniie_lab.dataclasses.instruction import (
    ClickInstruction,
    NextActionInstruction,
    QueryFormulationInstruction,
    QueryReFormulationInstruction,
    RelevanceJudgementInstruction,
)
from geniie_lab.memory import ConversationHistory
from geniie_lab.response import Clicks, NextAction, Query, RelevanceJudgement


class LLMServiceProtocol(Protocol):
    def create_query(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: QueryFormulationInstruction) -> Tuple[Query, int]:
        ...
    def create_clicks(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: ClickInstruction) -> Tuple[Clicks, int]:
        ...
    def recreate_query(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: QueryReFormulationInstruction) -> Tuple[Query, int]:
        ...
    def calc_relevance_judgement(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: RelevanceJudgementInstruction) -> Tuple[RelevanceJudgement, int]:
        ...
    def decide_next_action(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: NextActionInstruction) -> Tuple[NextAction, int]:
        ...
    def get_tokenizer(self, model_name: str) -> Callable[[str], int]: ...