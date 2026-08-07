from types import SimpleNamespace

from geniie_lab.dataclasses.description import ModelDescription
from geniie_lab.dataclasses.instruction import QueryReFormulationInstruction
from geniie_lab.memory import ConversationHistory
from geniie_lab.response import Query
from geniie_lab.services.llm.openai_compatible import OpenAICompatibleLLMService


class StubClient:
    """Captures the kwargs of beta.chat.completions.parse and returns a
    canned parsed completion."""

    def __init__(self, parsed):
        self.calls = []

        def parse(**kwargs):
            self.calls.append(kwargs)
            message = SimpleNamespace(parsed=parsed, to_json=lambda: '{"canned": true}')
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(total_tokens=99),
            )

        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        )


def test_call_preserves_roles_and_updates_memory():
    parsed = Query(query="q", start=0, size=10, reason="r")
    stub = StubClient(parsed)
    service = OpenAICompatibleLLMService(client=stub, tiktoken_by_model=False)

    memory = ConversationHistory(system_role=None, system_prompt="be helpful")
    memory.add_user_message("earlier user turn")
    memory.add_assistant_response("earlier assistant turn")

    instruction = QueryReFormulationInstruction(instruction="reformulate")
    model = ModelDescription(type="openai", name="some-model", token_length=100000,
                             temperature=0.0, top_p=1.0)
    response, total_token, thinking = service.generate(model, memory, instruction, Query)

    assert response is parsed
    assert total_token == 99
    # The stub message has no reasoning attributes: non-reasoning models
    # yield thinking=None.
    assert thinking is None

    # B4: roles must be passed through unchanged -- system prompt first, and
    # the earlier assistant turn kept as role=assistant.
    sent = stub.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[0]["content"] == "be helpful"
    assert sent[2] == {"role": "assistant", "content": "earlier assistant turn"}

    # Generation parameters are forwarded.
    assert stub.calls[0]["model"] == "some-model"
    assert stub.calls[0]["temperature"] == 0.0
    assert stub.calls[0]["response_format"] is Query

    # Memory gains the instruction (user) and the raw completion (assistant).
    history = memory.get_all_messages()
    assert history[-2]["role"] == "user"
    assert "reformulate" in history[-2]["content"]
    assert history[-1] == {"role": "assistant", "content": '{"canned": true}'}


class ReasoningStubClient(StubClient):
    """StubClient whose message carries reasoning fields, as returned by
    vLLM reasoning parsers and Bedrock for reasoning models."""

    def __init__(self, parsed, reasoning=None, reasoning_content=None):
        super().__init__(parsed)
        calls = self.calls

        def parse(**kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(
                parsed=parsed,
                reasoning=reasoning,
                reasoning_content=reasoning_content,
                to_json=lambda: '{"canned": true}',
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(total_tokens=99),
            )

        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        )


def _generate(stub, thinking_token_budget=None):
    service = OpenAICompatibleLLMService(client=stub, tiktoken_by_model=False)
    memory = ConversationHistory(system_role=None, system_prompt="be helpful")
    instruction = QueryReFormulationInstruction(instruction="reformulate")
    model = ModelDescription(type="vllm", name="some-model", token_length=100000,
                             thinking_token_budget=thinking_token_budget)
    return service.generate(model, memory, instruction, Query), stub.calls


def test_thinking_captured_from_reasoning_field():
    parsed = Query(query="q", start=0, size=10, reason="r")
    (response, _, thinking), _ = _generate(ReasoningStubClient(parsed, reasoning="chain of thought"))
    assert thinking == "chain of thought"
    assert response is parsed


def test_thinking_falls_back_to_reasoning_content():
    parsed = Query(query="q", start=0, size=10, reason="r")
    (_, _, thinking), _ = _generate(
        ReasoningStubClient(parsed, reasoning=None, reasoning_content="legacy field"))
    assert thinking == "legacy field"


def test_thinking_budget_forwarded_in_extra_body_only_when_set():
    parsed = Query(query="q", start=0, size=10, reason="r")
    _, calls = _generate(ReasoningStubClient(parsed), thinking_token_budget=256)
    assert calls[0]["extra_body"] == {"thinking_token_budget": 256}

    _, calls = _generate(ReasoningStubClient(parsed))
    assert "extra_body" not in calls[0]
