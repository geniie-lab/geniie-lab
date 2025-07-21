# Standard library
from typing import Callable, Protocol, Type, TypeVar

# Third-party libraries
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel
import tiktoken

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

T = TypeVar("T", bound=BaseModel)

class InstructionWithGenerate(Protocol):
    def generate(self) -> str:
        ...

class OpenAILLMService:
    _MAX_TOKEN_LIMITS = {
        "gpt-4o":             131072,  # GPT-4o large context
        "gpt-4o-mini":         8192,
        "gpt-4-turbo":         8192,
        "gpt-4.1-mini-2025-04-14": 1000000,
        "gpt-4.1-mini":        8192,
        "gpt-4":               8192,
        "gpt-4-32k":          32768,

        "gpt-3.5-turbo-16k":  16384,
        "gpt-3.5-turbo":       4096,

        "text-embedding-ada-002": 8191,
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
    }

    def __init__(self):
        self.client = OpenAI()

    def _call_llm_with_pydantic_response(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        instruction: InstructionWithGenerate,
        response_model: Type[T]
    ) -> T:

        memory.add_user_message(instruction.generate())
        messages_dicts: list[dict[str, str]] = memory.get_messages(tokenizer=self.get_tokenizer(model), max_tokens=self.get_max_tokens(model))
        messages: list[ChatCompletionUserMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=msg["content"]) for msg in messages_dicts
        ]
        completion = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
            temperature=temperature,
        )
        parsed_response = completion.choices[0].message.parsed
        if parsed_response is None:
            raise ValueError(f"LLM returned empty parsed object for {response_model.__name__}.")
        memory.add_assistant_response(completion.choices[0].message.to_json())

        return parsed_response

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]:

        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))

    def get_max_tokens(self, model_name: str) -> int:

        name = model_name.lower()
        for prefix, limit in self._MAX_TOKEN_LIMITS.items():
            if name.startswith(prefix):
                return limit
        return 4096

    def create_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryFormulationInstruction) -> Query:

        query = self._call_llm_with_pydantic_response(model, temperature, memory, instruction, Query)
        return query

    def recreate_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryReFormulationInstruction) -> Query:

        query = self._call_llm_with_pydantic_response(model, temperature, memory, instruction, Query)
        return query

    def create_clicks(self, model: str, temperature: float, memory: ConversationHistory, instruction: ClickInstruction) -> Clicks:

        return self._call_llm_with_pydantic_response(model, temperature, memory, instruction, Clicks)

    def calc_relevance_judgement(self, model: str, temperature: float, memory: ConversationHistory, instruction: RelevanceJudgementInstruction) -> RelevanceJudgement:

        return self._call_llm_with_pydantic_response(model, temperature, memory, instruction, RelevanceJudgement)

    def decide_next_action(self, model: str, temperature: float, memory: ConversationHistory, instruction: NextActionInstruction) -> NextAction:
        return self._call_llm_with_pydantic_response(model, temperature, memory, instruction, NextAction)
