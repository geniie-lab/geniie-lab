"""Visited-result marking: the SERP equivalent of a browser's visited links.

The fake LLM always clicks rank 1 (d1), so a plan with two search iterations
gives each topic one unmarked SERP and one where d1 should carry the marker.
"""
from geniie_lab.dataclasses.serp import SearchResultItem
from geniie_lab.experiments.agentic_experiment import ExperimentRunner as AgenticRunner
from geniie_lab.experiments.session_experiment import ExperimentRunner
from geniie_lab.response import Action

from tests.conftest import make_settings

TWO_ITERATIONS = ["query", "ranking", "click", "reformulate", "ranking", "click"]


def click_prompts(fake_llm, topic_index):
    """The rendered click prompts for one topic, in plan order."""
    prompts = [text for kind, text in fake_llm.prompts if kind == "ClickInstruction"]
    per_topic = len([s for s in TWO_ITERATIONS if s == "click"])
    return prompts[topic_index * per_topic:(topic_index + 1) * per_topic]


def test_unvisited_repr_is_unchanged_by_the_new_field():
    # Also guards score staying absent, which the hand-written repr could leak.
    item = SearchResultItem(ranking=1, docid="d1", title="Doc One",
                            snippet="about topic one", score=2.0)
    assert repr(item) == (
        "SearchResultItem(ranking=1, docid='d1', title='Doc One', "
        "snippet='about topic one')"
    )


def test_visited_repr_adds_only_the_marker():
    item = SearchResultItem(ranking=1, docid="d1", title="Doc One",
                            snippet="about topic one", score=2.0, visited=True)
    assert repr(item) == (
        "SearchResultItem(ranking=1, docid='d1', title='Doc One', "
        "snippet='about topic one', visited=True)"
    )


def test_visited_is_serialised_even_when_hidden_from_the_searcher():
    assert SearchResultItem(ranking=1, docid="d1", title="t",
                            snippet="s").to_dict()["visited"] is False


def test_clicked_document_is_marked_in_the_next_serp(patched_services):
    settings = make_settings(plan=TWO_ITERATIONS, mark_visited_results=True)
    ExperimentRunner(settings=settings).run()

    first, second = click_prompts(patched_services.llm, topic_index=0)
    assert "visited=True" not in first  # nothing opened yet
    assert "docid='d1', title='Doc One', snippet='about topic one', visited=True" in second
    assert "docid='d2'" in second and "visited" not in second.split("docid='d2'")[1]


def test_marks_do_not_leak_across_topics(patched_services):
    settings = make_settings(plan=TWO_ITERATIONS, mark_visited_results=True)
    ExperimentRunner(settings=settings).run()

    # t2 starts unmarked even though d1 was opened while working on t1.
    first_of_t2, second_of_t2 = click_prompts(patched_services.llm, topic_index=1)
    assert "visited=True" not in first_of_t2
    assert "visited=True" in second_of_t2


def test_flag_off_leaves_the_prompt_byte_identical(patched_services):
    settings = make_settings(plan=TWO_ITERATIONS, mark_visited_results=False)
    ExperimentRunner(settings=settings).run()

    first, second = click_prompts(patched_services.llm, topic_index=0)
    assert "visited" not in second
    assert first == second  # same SERP, and nothing marks it


def test_agentic_marks_a_reused_serp_without_a_new_ranking(patched_services):
    # CLICK_DOCUMENT runs no ranking stage, so the second click renders the
    # SERP object built for the first one. The marks must still be there.
    patched_services.llm.next_actions = [Action.CLICK_DOCUMENT, Action.END_TASK]
    settings = make_settings(name="test_agentic", max_actions=10,
                             mark_visited_results=True)
    AgenticRunner(settings=settings).run()

    prompts = [text for kind, text in patched_services.llm.prompts
               if kind == "ClickInstruction"]
    first, second = prompts[0], prompts[1]
    assert "visited=True" not in first
    assert "docid='d1', title='Doc One', snippet='about topic one', visited=True" in second
