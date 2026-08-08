import pytest
from pydantic import BaseModel, ValidationError

from geniie_lab.services.llm.json_repair import repair_and_validate


class Answer(BaseModel):
    crossings: int
    plan: str


def test_valid_json_passes_untouched():
    parsed, applied = repair_and_validate('{"crossings": 7, "plan": "go"}', Answer)
    assert parsed.crossings == 7
    assert applied == []


def test_strips_code_fences_and_prose():
    content = 'Here is the answer:\n```json\n{"crossings": 7, "plan": "go"}\n```\nDone.'
    parsed, applied = repair_and_validate(content, Answer)
    assert parsed.plan == "go"
    assert "extract_object" in applied


def test_fixes_trailing_escaped_quote():
    # Real-world shape observed from gpt-oss-120b on vLLM 0.17.
    content = '{"crossings": 7, "plan": "take the goat\\"}'
    parsed, applied = repair_and_validate(content, Answer)
    assert parsed.plan == "take the goat"
    assert "fix_trailing_escaped_quote" in applied


def test_closes_truncated_output():
    content = '{"crossings": 7, "plan": "take the goat'
    parsed, applied = repair_and_validate(content, Answer)
    assert parsed.plan == "take the goat"
    assert "close_unterminated" in applied


def test_escapes_raw_newlines_inside_strings():
    content = '{"crossings": 7, "plan": "1. goat\n2. return"}'
    parsed, applied = repair_and_validate(content, Answer)
    assert parsed.plan == "1. goat\n2. return"
    assert "escape_inner_newlines" in applied


def test_semantic_error_is_not_repairable():
    # Syntactically fine, semantically wrong: repair must not mask it.
    with pytest.raises(ValidationError):
        repair_and_validate('{"crossings": "many", "plan": 3}', Answer)


def test_no_json_at_all_raises():
    with pytest.raises((ValidationError, ValueError)):
        repair_and_validate('I cannot answer that.', Answer)


def test_determinism():
    content = 'noise {"crossings": 7, "plan": "go\\"}'
    parsed, applied = repair_and_validate(content, Answer)
    assert all(r == (parsed, applied) for r in
               [repair_and_validate(content, Answer) for _ in range(3)])
