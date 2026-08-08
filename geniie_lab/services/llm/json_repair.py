# Deterministic, rule-based repair of malformed JSON emitted by LLMs.
#
# Policy (issue #13, refined): repairs are pure functions applied in a fixed
# order, so a given raw output always yields the same repaired output — no
# model round-trip is involved at this layer. Callers report which rules
# fired; outputs that stay invalid after repair go to the caller's bounded
# feedback-retry loop, and topics whose calls exhaust it are skipped.

import json
import re
from typing import Callable, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) around the payload."""
    return re.sub(r"```[a-zA-Z]*", "", text)


def _extract_object(text: str) -> str:
    """Cut surrounding prose: keep from the first '{' to the last '}'.

    If no '}' exists at all (truncated output), keep from the first '{' to
    the end so later rules can close the object.
    """
    start = text.find("{")
    if start == -1:
        return text
    end = text.rfind("}")
    return text[start:end + 1] if end > start else text[start:]


def _escape_inner_newlines(text: str) -> str:
    """Escape raw newlines/tabs that appear inside JSON strings."""
    out, in_string, escaped = [], False, False
    for char in text:
        if in_string and char == "\n":
            out.append("\\n"); continue
        if in_string and char == "\t":
            out.append("\\t"); continue
        out.append(char)
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string
    return "".join(out)


def _fix_trailing_escaped_quote(text: str) -> str:
    """Fix a string that ends with an escaped quote where it should close.

    Observed in the wild (gpt-oss-120b on vLLM 0.17): '..."reason": "text\\"}'
    — the closing quote of the last string is escaped, so the string never
    terminates. Rewrite the final '\\"' before the closing brace to '"'.
    """
    return re.sub(r'\\"(\s*[}\]])', r'"\1', text)


def _close_unterminated(text: str) -> str:
    """Close an unterminated final string and any unbalanced braces/brackets
    (the signature of output truncated by a token limit)."""
    in_string, escaped = False, False
    stack = []
    for char in text:
        if escaped:
            escaped = False; continue
        if char == "\\":
            escaped = True; continue
        if char == '"':
            in_string = not in_string; continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()
    if in_string:
        text += '"'
    for opener in reversed(stack):
        text += "}" if opener == "{" else "]"
    return text


#: Fixed application order — part of the reproducibility contract.
REPAIR_RULES: list[Tuple[str, Callable[[str], str]]] = [
    ("strip_code_fences", _strip_code_fences),
    ("extract_object", _extract_object),
    ("escape_inner_newlines", _escape_inner_newlines),
    ("fix_trailing_escaped_quote", _fix_trailing_escaped_quote),
    ("close_unterminated", _close_unterminated),
]


def repair_and_validate(content: str, response_model: Type[T]) -> Tuple[T, list[str]]:
    """Validate content against response_model, repairing deterministically.

    Rules are applied cumulatively in REPAIR_RULES order, validating after
    each step; returns (parsed, applied_rule_names) where applied_rule_names
    covers every rule applied up to the first success ([] when the content
    was valid as-is). Raises the last ValidationError/ValueError if the
    content stays invalid after all rules — the caller decides what happens
    next (feedback retry, then skipping the topic).
    """
    applied: list[str] = []
    text = content
    last_error: Exception = ValueError(f"no JSON object found in: {content[:80]!r}")
    for name, rule in [("as_is", lambda t: t)] + REPAIR_RULES:
        if name != "as_is":
            repaired = rule(text)
            if repaired == text:
                continue
            text = repaired
            applied.append(name)
        try:
            json.loads(text)  # syntax first, for a precise error
            return response_model.model_validate_json(text), applied
        except (ValidationError, ValueError) as error:
            last_error = error
    raise last_error
