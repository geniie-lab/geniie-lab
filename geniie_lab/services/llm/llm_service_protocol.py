# Standard library
from typing import Callable, Protocol, Tuple, Type, TypeVar

# Third-party libraries
from pydantic import BaseModel

# Local application imports
from geniie_lab.dataclasses.description import ModelDescription
from geniie_lab.dataclasses.instruction import Instruction
from geniie_lab.memory import ConversationHistory

T = TypeVar("T", bound=BaseModel)


class LLMServiceProtocol(Protocol):
    def generate(self, model: ModelDescription, memory: ConversationHistory, instruction: Instruction, response_model: Type[T]) -> Tuple[T, int, str | None]:
        """Returns (parsed, total_token, thinking); thinking is the reasoning
        trace for reasoning models, None otherwise (capture only — never fed
        back into the conversation)."""
        ...

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]: ...
