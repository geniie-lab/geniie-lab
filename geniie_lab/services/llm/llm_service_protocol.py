# Standard library
from typing import Callable, Protocol

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
    def create_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryFormulationInstruction) -> Query:
        ...
    def create_clicks(self, model: str, temperature: float, memory: ConversationHistory, instruction: ClickInstruction) -> Clicks:
        ...
    def recreate_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryReFormulationInstruction) -> Query:
        ...
    def calc_relevance_judgement(self, model: str, temperature: float, memory: ConversationHistory, instruction: RelevanceJudgementInstruction) -> RelevanceJudgement:
        ...
    def decide_next_action(self, model: str, temperature: float, memory: ConversationHistory, instruction: NextActionInstruction) -> NextAction:
        ...
    def get_tokenizer(self, model_name: str) -> Callable[[str], int]: ...
    def get_max_tokens(self, model_name: str) -> int: ...