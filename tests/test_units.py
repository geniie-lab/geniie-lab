import math

import ir_measures
import pytest

from geniie_lab.dataclasses.description import (
    CorpusDescription,
    TaskDescription,
    ToolDescription,
)
from geniie_lab.dataclasses.instruction import (
    ClickInstruction,
    QueryFormulationInstruction,
    RelevanceJudgementInstruction,
)
from geniie_lab.dataclasses.serp import FullText, SearchResultItem, Serp
from geniie_lab.dataclasses.topic import FullTopic
from geniie_lab.memory import ConversationHistory
from geniie_lab.services.measure_service import MeasureService, Qrels, Run


# --- ConversationHistory -----------------------------------------------------

def test_memory_defaults_to_system_role():
    memory = ConversationHistory(system_role=None, system_prompt="be helpful")
    assert memory.get_all_messages() == [{"role": "system", "content": "be helpful"}]


def test_memory_prunes_oldest_messages_first():
    memory = ConversationHistory(system_role=None, system_prompt="sys")
    for i in range(5):
        memory.add_user_message(f"msg-{i}")

    # One token per message: budget of 3 keeps the system prompt plus the two
    # most recent messages, in chronological order.
    messages = memory.get_messages(tokenizer=lambda text: 1, max_tokens=3)
    assert [m["content"] for m in messages] == ["sys", "msg-3", "msg-4"]


def test_memory_clone_is_independent():
    memory = ConversationHistory(system_role=None, system_prompt="sys")
    memory.add_user_message("original")
    clone = memory.clone()
    clone.add_user_message("only in clone")

    assert len(memory.get_all_messages()) == 2
    assert len(clone.get_all_messages()) == 3


# --- MeasureService -----------------------------------------------------------

def test_measures_match_hand_computed_values():
    # d1 is relevant but scored below d2, so it lands at rank 2:
    # RR = 1/2, nDCG@10 = (1/log2(3)) / (1/log2(2)) = 0.6309...
    qrels = Qrels()
    qrels.add("t1", "d1", 1)
    run = Run()
    run.add("t1", "d1", 1.0)
    run.add("t1", "d2", 2.0)

    results = MeasureService().calc([ir_measures.nDCG@10, ir_measures.MRR@10], qrels, run)
    assert results["RR@10"] == pytest.approx(0.5)
    assert results["nDCG@10"] == pytest.approx(1 / math.log2(3))


# --- Instruction prompts --------------------------------------------------------

def _serp() -> Serp:
    return Serp(hits=2, results=[
        SearchResultItem(ranking=1, docid="d1", title="Doc One",
                         snippet="first snippet", score=12.5),
        SearchResultItem(ranking=2, docid="d2", title="Doc Two",
                         snippet="second snippet", score=3.25),
    ])


def test_query_formulation_prompt_contains_all_sections():
    prompt = QueryFormulationInstruction(
        instruction="Formulate a query.",
        task=TaskDescription(name="T", description="the task", measurement=[]),
        corpus=CorpusDescription(name="C", description="the corpus", index_name="idx"),
        tool=ToolDescription(name="opensearch", ranking_model="bm25",
                             index_name="idx", description="the tool"),
        topic=FullTopic(id="t1", title="the title", description="the description",
                        narrative="the narrative"),
    ).generate()

    for fragment in ["Formulate a query.", "the task", "the corpus", "the tool",
                     "the title", "the description", "the narrative"]:
        assert fragment in prompt


def test_click_prompt_shows_serp_but_not_retrieval_scores():
    # B1's disclosure guard: the simulated searcher sees titles and snippets,
    # never the engine's scores.
    prompt = ClickInstruction(instruction="Pick documents.", serp=_serp()).generate()

    assert "Doc One" in prompt and "first snippet" in prompt
    assert "score" not in prompt
    assert "12.5" not in prompt and "3.25" not in prompt


def test_relevance_prompt_contains_document_title_and_text():
    # B7: fetch_fulltext populates the title, and the prompt renders it.
    prompt = RelevanceJudgementInstruction(
        instruction="Judge relevance.",
        fulltext=FullText(docid="d1", text="the full text", title="the doc title"),
    ).generate()

    assert "the doc title" in prompt
    assert "the full text" in prompt
    assert "None" not in prompt
