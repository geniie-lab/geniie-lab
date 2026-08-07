# Standard library
import json
from typing import Callable, Tuple, Type, TypeVar

# Third-party libraries
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import tiktoken

# Local application imports
from geniie_lab.dataclasses.description import ModelDescription
from geniie_lab.dataclasses.instruction import Instruction
from geniie_lab.memory import ConversationHistory

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleLLMService:
    """One implementation for every provider that speaks the OpenAI chat API.

    Providers differ only in how the client is constructed (base_url/api_key,
    or a pre-built client such as AzureOpenAI) and in whether tiktoken should
    look up a model-specific encoding. The provider registry lives in
    llm_service_factory.
    """

    #: Attempts per call when schema_via_prompt validation fails.
    MAX_SCHEMA_RETRIES = 3

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: OpenAI | None = None,
        tiktoken_by_model: bool = True,
        schema_via_prompt: bool = False,
    ):
        self.client = client or OpenAI(base_url=base_url, api_key=api_key)
        self._tiktoken_by_model = tiktoken_by_model
        # Some providers (e.g. Amazon Bedrock) mis-handle json_schema
        # response_format for open-weight models: strict grammars loop on
        # whitespace and non-strict ones leak stray tokens before the JSON.
        # When set, send the schema in the prompt with json_object mode and
        # validate the response ourselves instead of using parse().
        self._schema_via_prompt = schema_via_prompt

    def generate(
        self,
        model: ModelDescription,
        memory: ConversationHistory,
        instruction: Instruction,
        response_model: Type[T],
    ) -> Tuple[T, int, str | None]:
        """Ask the model to respond to the instruction as a response_model.

        Returns (parsed, total_token, thinking): thinking is the model's
        reasoning trace when the provider exposes one (reasoning models served
        with a reasoning parser), None otherwise. It is captured for logging
        only and never fed back into the conversation history.
        """
        call = (self._call_llm_with_prompted_schema if self._schema_via_prompt
                else self._call_llm_with_pydantic_response)
        return call(
            model.name, model.token_length, model.temperature, model.top_p,
            memory, instruction, response_model,
            thinking_token_budget=model.thinking_token_budget,
        )

    @staticmethod
    def _extract_thinking(message) -> str | None:
        # vLLM (openai_gptoss/qwen3/nemotron parsers) and Bedrock all expose
        # the trace as message.reasoning; reasoning_content is the documented
        # older field name, kept as a fallback. Both absent on non-reasoning
        # models.
        return (getattr(message, "reasoning", None)
                or getattr(message, "reasoning_content", None))

    @staticmethod
    def _budget_kwargs(thinking_token_budget: int | None) -> dict:
        # vLLM enforces the cap server-side (qwen3/deepseek_r1 parsers);
        # providers that predate the field ignore it silently.
        if thinking_token_budget is None:
            return {}
        return {"extra_body": {"thinking_token_budget": thinking_token_budget}}

    def _call_llm_with_pydantic_response(
        self,
        model: str,
        token_length: int,
        temperature: float,
        top_p: float,
        memory: ConversationHistory,
        instruction: Instruction,
        response_model: Type[T],
        thinking_token_budget: int | None = None,
    ) -> Tuple[T, int, str | None]:

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
            **self._budget_kwargs(thinking_token_budget),
        )
        parsed_response = completion.choices[0].message.parsed
        if parsed_response is None:
            raise ValueError(f"LLM returned empty parsed object for {response_model.__name__}.")
        memory.add_assistant_response(completion.choices[0].message.to_json())

        usage = getattr(completion, "usage", None)
        total_token = getattr(usage, "total_tokens", 0) or 0

        return parsed_response, total_token, self._extract_thinking(completion.choices[0].message)

    def _call_llm_with_prompted_schema(
        self,
        model: str,
        token_length: int,
        temperature: float,
        top_p: float,
        memory: ConversationHistory,
        instruction: Instruction,
        response_model: Type[T],
        thinking_token_budget: int | None = None,
    ) -> Tuple[T, int, str | None]:
        tokenizer = self.get_tokenizer(model)
        schema_suffix = (
            "\n\nRespond with ONLY a JSON object that conforms to this JSON schema:\n"
            + json.dumps(response_model.model_json_schema())
        )
        # Store only the plain instruction in session memory so bedrock
        # conversations stay identical to providers that take the schema
        # out-of-band; the schema rides along on the outgoing copy only.
        memory.add_user_message(instruction.generate())
        messages: list[dict[str, str]] = memory.get_messages(tokenizer=tokenizer, max_tokens=token_length)
        messages = messages[:-1] + [
            {**messages[-1], "content": messages[-1]["content"] + schema_suffix}
        ]
        # The schema suffix exists only to work around the provider's broken
        # response_format; exclude it from the reported count so bedrock
        # sessions are token-comparable with other providers. The deduction
        # uses the trimming tokenizer, so it is approximate.
        schema_tokens = tokenizer(schema_suffix)
        total_token = 0
        last_error: Exception | None = None
        for _ in range(self.MAX_SCHEMA_RETRIES):
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p,
                **self._budget_kwargs(thinking_token_budget),
            )
            usage = getattr(completion, "usage", None)
            call_tokens = getattr(usage, "total_tokens", 0) or 0
            total_token += max(0, call_tokens - schema_tokens)
            content = completion.choices[0].message.content or ""
            try:
                parsed_response = self._parse_lenient(content, response_model)
            except (ValidationError, ValueError) as error:
                # Feedback retry: keep the failed exchange out of the session
                # memory but show it to the model so the next attempt differs.
                last_error = error
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content":
                        f"Your response was not valid: {error}. "
                        "Respond again with ONLY a valid JSON object for the schema."},
                ]
                continue
            memory.add_assistant_response(completion.choices[0].message.to_json())
            return parsed_response, total_token, self._extract_thinking(completion.choices[0].message)
        raise ValueError(
            f"LLM response failed {response_model.__name__} validation after "
            f"{self.MAX_SCHEMA_RETRIES} attempts: {last_error}"
        )

    @staticmethod
    def _parse_lenient(content: str, response_model: Type[T]) -> T:
        """Validate content, tolerating stray tokens around the JSON object.

        Providers without a real schema grammar can emit prefixes such as
        "1,{...}" or "{ {...}" around an otherwise valid object; try each
        "{"-suffix of the content until one validates.
        """
        end = content.rfind("}") + 1
        starts = [i for i, char in enumerate(content[:end]) if char == "{"]
        last_error: Exception = ValueError(f"no JSON object found in: {content[:80]!r}")
        for start in starts:
            try:
                return response_model.model_validate_json(content[start:end])
            except ValidationError as error:
                last_error = error
        raise last_error

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]:
        if self._tiktoken_by_model:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
