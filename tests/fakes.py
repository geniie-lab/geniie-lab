# Test doubles for the geniie-lab service protocols.
#
# FakeLLMService mimics the real services' observable contract: it appends the
# generated instruction as a user message and its response as an assistant
# message to the ConversationHistory, and returns (response, total_token).
import json
from collections import namedtuple
from typing import List

from geniie_lab.dataclasses.serp import FullText, SearchResultItem, Serp
from geniie_lab.response import (
    Action,
    Clicks,
    NextAction,
    Query,
    Relevance,
    RelevanceJudgement,
)

FAKE_TOTAL_TOKEN = 42

FakeQuery = namedtuple("FakeQuery", ["query_id", "title", "description", "narrative"])
FakeQrel = namedtuple("FakeQrel", ["query_id", "doc_id", "relevance"])


class FakeDataset:
    """Stands in for an ir_datasets dataset (queries_iter / qrels_iter)."""

    def __init__(self, queries: List[FakeQuery], qrels: List[FakeQrel]):
        self._queries = queries
        self._qrels = qrels

    def queries_iter(self):
        return iter(self._queries)

    def qrels_iter(self):
        return iter(self._qrels)


def default_dataset() -> FakeDataset:
    return FakeDataset(
        queries=[
            FakeQuery("t1", "topic one", "description one", "narrative one"),
            FakeQuery("t2", "topic two", "description two", "narrative two"),
        ],
        # Only d1 is relevant, and only for t1. The fake SERP always returns
        # d1 at rank 1 with the highest score, so t1 must score RR@10 = 1.0.
        qrels=[FakeQrel("t1", "d1", 1)],
    )


class FakeLLMService:
    """Implements LLMServiceProtocol with canned responses.

    `calls` records (method_name, history_len_before_call) so tests can assert
    memory-cloning semantics. `next_actions` is a FIFO of Action values for
    decide_next_action; when exhausted it returns END_TASK.
    """

    def __init__(self, next_actions: List[Action] | None = None):
        self.calls: list[tuple[str, int]] = []
        self.next_actions: list[Action] = list(next_actions or [])

    def _respond(self, method, memory, instruction, response):
        self.calls.append((method, len(memory._history)))
        memory.add_user_message(instruction.generate())
        memory.add_assistant_response(response.model_dump_json())
        return response, FAKE_TOTAL_TOKEN

    def create_query(self, model, token_length, temperature, top_p, memory, instruction):
        return self._respond(
            "create_query", memory, instruction,
            Query(query="first query", start=0, size=10, reason="initial query"),
        )

    def recreate_query(self, model, token_length, temperature, top_p, memory, instruction):
        return self._respond(
            "recreate_query", memory, instruction,
            Query(query="reformulated query", start=0, size=10, reason="try again"),
        )

    def create_clicks(self, model, token_length, temperature, top_p, memory, instruction):
        return self._respond(
            "create_clicks", memory, instruction,
            Clicks(ranking_list=[1], reason="top result looks relevant"),
        )

    def calc_relevance_judgement(self, model, token_length, temperature, top_p, memory, instruction):
        return self._respond(
            "calc_relevance_judgement", memory, instruction,
            RelevanceJudgement(label=Relevance.RELEVANT, reason="on topic"),
        )

    def decide_next_action(self, model, token_length, temperature, top_p, memory, instruction):
        action = self.next_actions.pop(0) if self.next_actions else Action.END_TASK
        return self._respond(
            "decide_next_action", memory, instruction,
            NextAction(action=action, reason="fake decision"),
        )

    def get_tokenizer(self, model_name):
        return lambda text: len(text)


class FakeOpenSearchClient:
    """Implements OpenSearchClientProtocol with a fixed two-document SERP.

    d1 always ranks first with the highest score. `searches` records
    (query, start, size) so tests can assert pagination behavior.
    """

    def __init__(self):
        self.searches: list[tuple[str, int, int]] = []

    def clean_text(self, text: str) -> str:
        return text

    def search_index_with_snippets(self, query: str, start: int = 0, size: int = 10) -> Serp:
        self.searches.append((query, start, size))
        return Serp(hits=2, results=[
            SearchResultItem(ranking=start + 1, docid="d1", title="Doc One",
                             snippet="about topic one", score=2.0),
            SearchResultItem(ranking=start + 2, docid="d2", title="Doc Two",
                             snippet="about something else", score=1.0),
        ])

    def fetch_fulltext(self, docid: str) -> FullText:
        return FullText(docid=docid, text=f"Full text of {docid}.", title=f"Title of {docid}")


def parse_jsonl(text: str) -> list[dict]:
    """Parse the experiment's stdout data channel into a list of records."""
    return [json.loads(line) for line in text.strip().splitlines() if line.strip()]
