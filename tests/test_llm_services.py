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
    message contents, one per call (the last repeats)."""

    def __init__(self, *contents):
        self.calls = []
        self._contents = list(contents)

        def create(**kwargs):
            self.calls.append(kwargs)
            content = self._contents.pop(0) if len(self._contents) > 1 else self._contents[0]
            message = SimpleNamespace(content=content, to_json=lambda: '{"canned": true}')
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(total_tokens=99),
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _service_call(stub, **service_kwargs):
    service = OpenAICompatibleLLMService(client=stub, tiktoken_by_model=False, **service_kwargs)
    memory = ConversationHistory(system_role=None, system_prompt="be helpful")
    memory.add_user_message("earlier user turn")
    memory.add_assistant_response("earlier assistant turn")
    instruction = QueryReFormulationInstruction(instruction="reformulate")
    model = ModelDescription(type="openai", name="some-model", token_length=100000,
                             temperature=0.0, top_p=1.0)
    return service.generate(model, memory, instruction, Query), memory


def test_call_preserves_roles_and_updates_memory():
    stub = StubClient(VALID_QUERY_JSON)
    (response, total_token), memory = _service_call(stub)

    assert response == Query(query="q", start=0, size=10, reason="r")
    assert total_token == 99

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

    # Memory gains the instruction (user) and the raw completion (assistant).
    history = memory.get_all_messages()
    assert history[-2]["role"] == "user"
    assert "reformulate" in history[-2]["content"]
    assert history[-1] == {"role": "assistant", "content": '{"canned": true}'}


def test_schema_via_prompt_appends_schema_to_outgoing_copy_only():
    stub = StubClient(VALID_QUERY_JSON)
    (response, _), memory = _service_call(stub, schema_via_prompt=True)

    assert response.query == "q"
    assert stub.calls[0]["response_format"] == {"type": "json_object"}
    # Schema rides on the outgoing message; session memory keeps the plain one.
    assert "JSON schema" in stub.calls[0]["messages"][-1]["content"]
    assert "JSON schema" not in memory.get_all_messages()[-2]["content"]


def test_repaired_output_needs_no_second_call():
    # Real failure shape (gpt-oss on vLLM 0.17): escaped closing quote.
    broken = '{"query": "q", "start": 0, "size": 10, "reason": "big fish\\"}'
    stub = StubClient(broken)
    (response, total_token), _ = _service_call(stub)

    assert response.reason == "big fish"
    assert len(stub.calls) == 1  # rule-based repair, no model round-trip
    assert total_token == 99


def test_feedback_retry_after_unrepairable_output():
    stub = StubClient('not json at all', VALID_QUERY_JSON)
    (response, total_token), _ = _service_call(stub)

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
