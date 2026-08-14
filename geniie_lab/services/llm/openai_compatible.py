# Standard library
import json
import re
import sys
from typing import Callable, Tuple, Type, TypeVar

# Third-party libraries
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import tiktoken

# Local application imports
from geniie_lab.dataclasses.description import ModelDescription
from geniie_lab.dataclasses.instruction import Instruction
from geniie_lab.memory import ConversationHistory
from geniie_lab.services.llm.json_repair import repair_and_validate

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleLLMService:
    """One implementation for every provider that speaks the OpenAI chat API.

    Providers differ only in how the client is constructed (base_url/api_key,
    or a pre-built client such as AzureOpenAI) and in whether tiktoken should
    look up a model-specific encoding. The provider registry lives in
    llm_service_factory.

    Invalid structured output is handled uniformly (issue #13): deterministic
    rule-based repair first (json_repair module), then a bounded
    validation-feedback retry, then a ValueError that the experiment runners
    turn into skipping the topic. Repairs and retries are reported on stderr
    so every recovery is visible in the run log.
    """

    #: Attempts per call when validation fails after repair.
    MAX_SCHEMA_RETRIES = 3

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: OpenAI | None = None,
        tiktoken_by_model: bool = True,
        schema_in_prompt: bool = False,
        json_object_fallback: bool = False,
    ):
        self.client = client or OpenAI(base_url=base_url, api_key=api_key)
        self._tiktoken_by_model = tiktoken_by_model
        # Field titles and descriptions reach the model either way, but out of
        # band in response_format the model attends to them less. Off by
        # default: it changes every prompt, and duplicates the schema on the
        # wire at the cost of the tokens.
        self._schema_in_prompt = schema_in_prompt
        # For providers that mis-handle json_schema response_format (Amazon
        # Bedrock with open-weight models: strict grammars loop on whitespace,
        # non-strict ones leak stray tokens before the JSON). Drops grammar
        # enforcement, leaving the repair-and-retry path to catch bad output,
        # so it needs schema_in_prompt to tell the model what to produce.
        self._json_object_fallback = json_object_fallback

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
        return self._call_llm_with_repair(
            model.name, model.token_length, model.temperature, model.top_p,
            memory, instruction, response_model,
            thinking_kwargs=self._thinking_kwargs(model),
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
    def _thinking_kwargs(model: ModelDescription) -> dict:
        # Thinking-length controls, forwarded only when set so providers
        # without them see an unchanged request:
        # - thinking_token_budget: server-side cap (vLLM qwen3/deepseek_r1
        #   parsers); silently ignored elsewhere.
        # - reasoning_effort: trained-in dial (gpt-oss low/medium/high).
        extra_body = {}
        if model.thinking_token_budget is not None:
            extra_body["thinking_token_budget"] = model.thinking_token_budget
        if model.reasoning_effort is not None:
            extra_body["reasoning_effort"] = model.reasoning_effort
        return {"extra_body": extra_body} if extra_body else {}

    def _call_llm_with_repair(
        self,
        model: str,
        token_length: int,
        temperature: float,
        top_p: float,
        memory: ConversationHistory,
        instruction: Instruction,
        response_model: Type[T],
        thinking_kwargs: dict = {},
    ) -> Tuple[T, int, str | None]:
        tokenizer = self.get_tokenizer(model)

        # Store only the plain instruction in session memory: the schema rides
        # along on the outgoing copy only, so conversations stay identical
        # across providers.
        memory.add_user_message(instruction.generate())
        # Pass roles through unchanged: the system prompt and previous
        # assistant turns must not be re-sent as user messages.
        messages: list[dict[str, str]] = memory.get_messages(tokenizer=tokenizer, max_tokens=token_length)

        schema_tokens = 0
        if self._schema_in_prompt:
            # Appended to the outgoing copy only, so session memory stays
            # identical across providers and conditions.
            schema_suffix = (
                "\n\nRespond with ONLY a JSON object that conforms to this JSON schema:\n"
                + json.dumps(response_model.model_json_schema())
            )
            messages = messages[:-1] + [
                {**messages[-1], "content": messages[-1]["content"] + schema_suffix}
            ]
            if self._json_object_fallback:
                # Here the suffix is forced by the provider's broken
                # response_format, so exclude it: the session stays
                # token-comparable with providers that send the schema out of
                # band. Approximate (trimming tokenizer). Requested as an
                # experimental condition instead, it is a real cost and counts.
                schema_tokens = tokenizer(schema_suffix)

        if self._json_object_fallback:
            response_format = {"type": "json_object"}
        else:
            # strict matches what the SDK's parse() helper sent before the
            # repair rewrite, so grammar enforcement stays as tight as before.
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    # Providers require ^[a-zA-Z0-9_-]+$; a parametrised
                    # generic's __name__ carries brackets, e.g.
                    # SubtopicRelevanceJudgement[RubricRelevance].
                    "name": re.sub(r"[^a-zA-Z0-9_-]", "_", response_model.__name__),
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            }

        total_token = 0
        last_error: Exception | None = None
        for attempt in range(self.MAX_SCHEMA_RETRIES):
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                top_p=top_p,
                **thinking_kwargs,
            )
            usage = getattr(completion, "usage", None)
            call_tokens = getattr(usage, "total_tokens", 0) or 0
            total_token += max(0, call_tokens - schema_tokens)
            message = completion.choices[0].message
            content = message.content or ""
            try:
                parsed_response, repairs = repair_and_validate(content, response_model)
            except (ValidationError, ValueError) as error:
                # Bounded feedback retry: keep the failed exchange out of the
                # session memory but show it to the model so the next attempt
                # differs. Reported so the recovery is visible in the run log.
                last_error = error
                print(f"[json-repair] attempt {attempt + 1}/{self.MAX_SCHEMA_RETRIES} "
                      f"invalid {response_model.__name__}: {error}", file=sys.stderr)
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content":
                        f"Your response was not valid: {error}. "
                        "Respond again with ONLY a valid JSON object for the schema."},
                ]
                continue
            if repairs:
                print(f"[json-repair] repaired {response_model.__name__} "
                      f"with rules {repairs}", file=sys.stderr)
            # Store the assistant's content only — not the serialized message
            # envelope, which on reasoning-parser deployments includes the
            # full reasoning trace and would silently feed it back (and
            # re-bill it) on every subsequent call. The stated `reason` field
            # inside the content is the rationale the session remembers;
            # thinking stays capture-only.
            memory.add_assistant_response(message.content or "")
            return parsed_response, total_token, self._extract_thinking(message)
        raise ValueError(
            f"LLM response failed {response_model.__name__} validation after "
            f"{self.MAX_SCHEMA_RETRIES} attempts: {last_error}"
        )

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]:
        if self._tiktoken_by_model:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
