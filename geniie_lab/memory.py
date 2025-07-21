import sys
from typing import List, Dict, Callable

class ConversationHistory:
    def __init__(self, system_role: str | None, system_prompt: str):
        if system_role is None:
            system_role = "system"
        self._system_prompt = {"role": system_role, "content": system_prompt}
        self._history: List[Dict] = []

    def add_user_message(self, content: str):
        self._history.append({"role": "user", "content": content})

    def add_assistant_response(self, response_content: str):
        self._history.append({"role": "assistant", "content": response_content})

    def remove_last_message(self):
        if self._history:
            self._history.pop()
            
    def get_messages(self, tokenizer: Callable[[str], int], max_tokens: int) -> List[Dict[str, str]]:
        messages = [self._system_prompt]

        current_tokens = tokenizer(self._system_prompt["content"])

        for message in reversed(self._history):
            message_tokens = tokenizer(message["content"])
            if current_tokens + message_tokens > max_tokens:
                print("Context window limit reached. Pruning older messages.", file=sys.stderr)
                break

            messages.insert(1, message)
            current_tokens += message_tokens

        return messages

    def get_all_messages(self) -> List[Dict[str, str]]:
        return [self._system_prompt] + self._history

    def clone(self) -> "ConversationHistory":
        from copy import deepcopy
        cloned = ConversationHistory(system_role=self._system_prompt["role"], system_prompt=self._system_prompt["content"])
        cloned._history = deepcopy(self._history)
        return cloned

