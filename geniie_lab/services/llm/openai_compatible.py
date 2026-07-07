# Standard library
from typing import Callable, Tuple, Type, TypeVar

# Third-party libraries
from openai import OpenAI
from pydantic import BaseModel
import tiktoken

# Local application imports
from geniie_lab.dataclasses.description import ModelDescription
from geniie_lab.dataclasses.instruction import (
    ClickInstruction,
    Instruction,
    NextActionInstruction,
    QueryFormulationInstruction,
    QueryReFormulationInstruction,
    RelevanceJudgementInstruction,
)
from geniie_lab.memory import ConversationHistory
from geniie_lab.response import Clicks, NextAction, Query, RelevanceJudgement

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleLLMService:
    """One implementation for every provider that speaks the OpenAI chat API.

    Providers differ only in how the client is constructed (base_url/api_key,
    or a pre-built client such as AzureOpenAI) and in whether tiktoken should
    look up a model-specific encoding. The provider registry lives in
    llm_service_factory.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: OpenAI | None = None,
        tiktoken_by_model: bool = True,
    ):
        self.client = client or OpenAI(base_url=base_url, api_key=api_key)
        self._tiktoken_by_model = tiktoken_by_model

    def generate(
        self,
        model: ModelDescription,
        memory: ConversationHistory,
        instruction: Instruction,
        response_model: Type[T],
    ) -> Tuple[T, int]:
        """Ask the model to respond to the instruction as a response_model."""
        return self._call_llm_with_pydantic_response(
            model.name, model.token_length, model.temperature, model.top_p,
            memory, instruction, response_model,
        )

    def _call_llm_with_pydantic_response(
        self,
        model: str,
        token_length: int,
        temperature: float,
        top_p: float,
        memory: ConversationHistory,
        instruction: Instruction,
        response_model: Type[T]
    ) -> Tuple[T, int]:

        memory.add_user_message(instruction.generate())
        # Pass roles through unchanged: the system prompt and previous assistant
        # turns must not be re-sent as user messages.
        messages: list[dict[str, str]] = memory.get_messages(tokenizer=self.get_tokenizer(model), max_tokens=token_length)
        completion = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
            temperature=temperature,
            top_p=top_p,
        )
        parsed_response = completion.choices[0].message.parsed
        if parsed_response is None:
            raise ValueError(f"LLM returned empty parsed object for {response_model.__name__}.")
        memory.add_assistant_response(completion.choices[0].message.to_json())

        usage = getattr(completion, "usage", None)
        total_token = getattr(usage, "total_tokens", 0) or 0

        return parsed_response, total_token

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]:
        if self._tiktoken_by_model:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))

    def create_query(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: QueryFormulationInstruction) -> Tuple[Query, int]:
        return self._call_llm_with_pydantic_response(model, token_length, temperature, top_p, memory, instruction, Query)

    def recreate_query(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: QueryReFormulationInstruction) -> Tuple[Query, int]:
        return self._call_llm_with_pydantic_response(model, token_length, temperature, top_p, memory, instruction, Query)

    def create_clicks(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: ClickInstruction) -> Tuple[Clicks, int]:
        return self._call_llm_with_pydantic_response(model, token_length, temperature, top_p, memory, instruction, Clicks)

    def calc_relevance_judgement(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: RelevanceJudgementInstruction) -> Tuple[RelevanceJudgement, int]:
        return self._call_llm_with_pydantic_response(model, token_length, temperature, top_p, memory, instruction, RelevanceJudgement)

    def decide_next_action(self, model: str, token_length: int, temperature: float, top_p: float, memory: ConversationHistory, instruction: NextActionInstruction) -> Tuple[NextAction, int]:
        return self._call_llm_with_pydantic_response(model, token_length, temperature, top_p, memory, instruction, NextAction)
