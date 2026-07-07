from geniie_lab.experiments.agentic_experiment import ExperimentRunner
from geniie_lab.response import Action

from tests.conftest import make_settings
from tests.fakes import parse_jsonl

# The bootstrap action is SUBMIT_NEW_QUERY, which maps to this stage cycle.
CYCLE = ["query", "ranking", "click", "rel_judge", "next_action"]


def test_end_task_ends_topic_not_experiment(patched_services, capsys):
    # B3: the fake decides END_TASK on the first decision of every topic;
    # topic t2 must still run after t1 ends its task.
    settings = make_settings(name="test_agentic", max_actions=10)
    ExperimentRunner(settings=settings).run()
    records = parse_jsonl(capsys.readouterr().out)

    t1 = [r["stage"] for r in records if r["topic_id"] == "t1"]
    t2 = [r["stage"] for r in records if r["topic_id"] == "t2"]
    assert t1 == CYCLE
    assert t2 == CYCLE  # pre-B3, t2 was silently skipped

    end = next(r for r in records if r["stage"] == "next_action")
    assert end["action"] == "END_TASK"


def test_go_next_result_page_paginates_by_serp_size(patched_services, capsys):
    # B5+B6: GO_NEXT_RESULT_PAGE advances the search offset by serp_size and
    # the ranking record logs the offset actually used.
    patched_services.llm.next_actions = [Action.GO_NEXT_RESULT_PAGE, Action.END_TASK]
    settings = make_settings(max_actions=20)
    ExperimentRunner(settings=settings).run()
    records = parse_jsonl(capsys.readouterr().out)

    t1_searches = patched_services.search.searches[:2]
    assert t1_searches == [("first query", 0, 10), ("first query", 10, 10)]

    t1_ranking_starts = [r["start"] for r in records
                         if r["topic_id"] == "t1" and r["stage"] == "ranking"]
    assert t1_ranking_starts == [0, 10]

    # GO_NEXT_RESULT_PAGE skips the query stage: ranking, click, relevance, decide.
    t1 = [r["stage"] for r in records if r["topic_id"] == "t1"]
    assert t1 == CYCLE + ["ranking", "click", "rel_judge", "next_action"]


def test_max_actions_terminates_the_loop(patched_services, capsys):
    # The fake keeps requesting new queries; the action budget must stop it.
    patched_services.llm.next_actions = [Action.SUBMIT_NEW_QUERY] * 50
    settings = make_settings(max_actions=7)
    ExperimentRunner(settings=settings).run()
    records = parse_jsonl(capsys.readouterr().out)

    # action_num is checked at the start of each cycle (1 and 6 pass, 11 does
    # not), so exactly two full cycles run per topic.
    t1 = [r["stage"] for r in records if r["topic_id"] == "t1"]
    assert t1 == CYCLE * 2
