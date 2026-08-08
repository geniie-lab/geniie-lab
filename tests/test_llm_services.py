from types import SimpleNamespace

import pytest

from geniie_lab.dataclasses.description import ModelDescription
from geniie_lab.dataclasses.instruction import QueryReFormulationInstruction
from geniie_lab.memory import ConversationHistory
from geniie_lab.response import Query
from geniie_lab.services.llm.openai_compatible import OpenAICompatibleLLMService

VALID_QUERY_JSON = '{"query": "q", "start": 0, "size": 10, "reason": "r"}'


class StubClient:
    """Captures the kwargs of chat.completions.create and returns canned
    message contents, one per call (the last repeats). Reasoning-model
    behaviour is stubbed via the reasoning/reasoning_content kwargs."""

    def __init__(self, *contents, reasoning=None, reasoning_content=None):
        self.calls = []
        self._contents = list(contents)
        self._reasoning = reasoning
        self._reasoning_content = reasoning_content

        def create(**kwargs):
            self.calls.append(kwargs)
            content = self._contents.pop(0) if len(self._contents) > 1 else self._contents[0]
            message = SimpleNamespace(content=content,
                                      reasoning=self._reasoning,
                                      reasoning_content=self._reasoning_content,
                                      to_json=lambda: '{"canned": true}')
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(total_tokens=99),
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _service_call(stub, thinking_token_budget=None, reasoning_effort=None, **service_kwargs):
    service = OpenAICompatibleLLMService(client=stub, tiktoken_by_model=False, **service_kwargs)
    memory = ConversationHistory(system_role=None, system_prompt="be helpful")
    memory.add_user_message("earlier user turn")
    memory.add_assistant_response("earlier assistant turn")
    instruction = QueryReFormulationInstruction(instruction="reformulate")
    model = ModelDescription(type="openai", name="some-model", token_length=100000,
                             temperature=0.0, top_p=1.0,
                             thinking_token_budget=thinking_token_budget,
                             reasoning_effort=reasoning_effort)
    return service.generate(model, memory, instruction, Query), memory


def test_call_preserves_roles_and_updates_memory():
    stub = StubClient(VALID_QUERY_JSON)
    (response, total_token, thinking), memory = _service_call(stub)

    assert response == Query(query="q", start=0, size=10, reason="r")
    assert total_token == 99
    # The stub message has no reasoning set: non-reasoning models yield None.
    assert thinking is None

    # B4: roles must be passed through unchanged -- system prompt first, and
    # the earlier assistant turn kept as role=assistant.
    sent = stub.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[0]["content"] == "be helpful"
    assert sent[2] == {"role": "assistant", "content": "earlier assistant turn"}

    # Generation parameters and the out-of-band schema are forwarded.
    assert stub.calls[0]["model"] == "some-model"
    assert stub.calls[0]["temperature"] == 0.0
    assert stub.calls[0]["response_format"]["type"] == "json_schema"
    assert stub.calls[0]["response_format"]["json_schema"]["name"] == "Query"

    # Memory gains the instruction (user) and the assistant's CONTENT — not
    # the serialized message envelope, which would leak the reasoning trace
    # into subsequent turns on reasoning-parser deployments.
    history = memory.get_all_messages()
    assert history[-2]["role"] == "user"
    assert "reformulate" in history[-2]["content"]
    assert history[-1] == {"role": "assistant", "content": VALID_QUERY_JSON}


def test_memory_never_contains_the_thinking_trace():
    stub = StubClient(VALID_QUERY_JSON, reasoning="chain of thought")
    (_, _, thinking), memory = _service_call(stub)
    assert thinking == "chain of thought"
    stored = " ".join(m["content"] for m in memory.get_all_messages())
    assert "chain of thought" not in stored
    # The stated reason DOES persist in memory, for all model kinds.
    assert '"reason": "r"' in stored


def test_schema_via_prompt_appends_schema_to_outgoing_copy_only():
    stub = StubClient(VALID_QUERY_JSON)
    (response, _, _), memory = _service_call(stub, schema_via_prompt=True)

    assert response.query == "q"
    assert stub.calls[0]["response_format"] == {"type": "json_object"}
    # Schema rides on the outgoing message; session memory keeps the plain one.
    assert "JSON schema" in stub.calls[0]["messages"][-1]["content"]
    assert "JSON schema" not in memory.get_all_messages()[-2]["content"]


def test_repaired_output_needs_no_second_call():
    # Real failure shape (gpt-oss on vLLM 0.17): escaped closing quote.
    broken = '{"query": "q", "start": 0, "size": 10, "reason": "big fish\\"}'
    stub = StubClient(broken)
    (response, total_token, _), _ = _service_call(stub)

    assert response.reason == "big fish"
    assert len(stub.calls) == 1  # rule-based repair, no model round-trip
    assert total_token == 99


def test_feedback_retry_after_unrepairable_output():
    stub = StubClient('not json at all', VALID_QUERY_JSON)
    (response, total_token, _), _ = _service_call(stub)

    assert response.query == "q"
    assert len(stub.calls) == 2
    # The retry conversation contains the failed output and the feedback.
    retry_messages = stub.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "not json at all"}
    assert "not valid" in retry_messages[-1]["content"]
    # Tokens from both attempts are accounted.
    assert total_token == 198


def test_exhausted_retries_raise_for_topic_skip():
    stub = StubClient('not json at all')
    with pytest.raises(ValueError, match="after 3 attempts"):
        _service_call(stub)
    assert len(stub.calls) == 3


def test_thinking_captured_from_reasoning_field():
    stub = StubClient(VALID_QUERY_JSON, reasoning="chain of thought")
    (response, _, thinking), _ = _service_call(stub)
    assert thinking == "chain of thought"
    assert response.query == "q"


def test_thinking_falls_back_to_reasoning_content():
    stub = StubClient(VALID_QUERY_JSON, reasoning=None, reasoning_content="legacy field")
    (_, _, thinking), _ = _service_call(stub)
    assert thinking == "legacy field"


def test_thinking_controls_forwarded_in_extra_body_only_when_set():
    stub = StubClient(VALID_QUERY_JSON)
    _service_call(stub, thinking_token_budget=256, reasoning_effort="high")
    assert stub.calls[0]["extra_body"] == {"thinking_token_budget": 256,
                                          "reasoning_effort": "high"}

    stub = StubClient(VALID_QUERY_JSON)
    _service_call(stub)
    assert "extra_body" not in stub.calls[0]
