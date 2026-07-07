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
    response, total_token = service.generate(model, memory, instruction, Query)

    assert response is parsed
    assert total_token == 99

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
