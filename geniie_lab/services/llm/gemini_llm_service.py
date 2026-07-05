# Standard library
import json
import os
from typing import Any, Callable, Dict, List, Protocol, Tuple, Type, TypeVar

# Third-party libraries
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

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

class GeminiLLMService:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _call_llm_and_parse(
        self,
        model: str,
        token_length: int,
        temperature: float,
        memory: ConversationHistory,
        instruction: InstructionWithGenerate,
        response_model: Type[T]
    ) -> Tuple[T, int]:

        memory.add_user_message(instruction.generate())
        openai_messages = memory.get_messages(
            tokenizer=self.get_tokenizer(model),
            max_tokens=token_length
        )

        system_prompt = openai_messages[0]['content'] if openai_messages and openai_messages[0]['role'] == 'system' else None
        gemini_contents: List[Dict[str, Any]] = []
        for msg in openai_messages[1:]:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        response = self.client.models.generate_content(
            model=model,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        if response.text is None:
            raise ValueError(f"Response text is None for {response_model.__name__}.")
        memory.add_assistant_response(response.text)
        data = json.loads(response.text)

        usage = getattr(response, "usage_metadata", None)
        total_token = getattr(usage, "total_token_count", 0) or 0

        return response_model(**data), total_token

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]:
        def count_fn(text: str) -> int:
            resp = self.client.models.count_tokens(model=model_name, contents=text)
            token_count = getattr(resp, "total_tokens", None)
            if token_count is None:
                raise ValueError(f"Token count is None for model {model_name}.")
            return token_count
        return count_fn

    def create_query(self, model: str, token_length: int, temperature: float, memory: ConversationHistory, instruction: QueryFormulationInstruction ) -> Tuple[Query, int]:

        query = self._call_llm_and_parse(model, token_length, temperature, memory, instruction, Query)
        return query

    def recreate_query(self, model: str, token_length: int, temperature: float, memory: ConversationHistory, instruction: QueryReFormulationInstruction) -> Tuple[Query, int]:

        query = self._call_llm_and_parse(model, token_length, temperature, memory, instruction, Query)
        return query

    def create_clicks(self, model: str, token_length: int, temperature: float, memory: ConversationHistory, instruction: ClickInstruction) -> Tuple[Clicks, int]:

        return self._call_llm_and_parse(model, token_length, temperature, memory, instruction, Clicks)

    def calc_relevance_judgement(self, model: str, token_length: int, temperature: float, memory: ConversationHistory, instruction: RelevanceJudgementInstruction) -> Tuple[RelevanceJudgement, int]:

        return self._call_llm_and_parse(model, token_length, temperature, memory, instruction, RelevanceJudgement)

    def decide_next_action(self, model: str, token_length: int, temperature: float, memory: ConversationHistory, instruction: NextActionInstruction) -> Tuple[NextAction, int]:
        return self._call_llm_and_parse(model, token_length, temperature, memory, instruction, NextAction)
