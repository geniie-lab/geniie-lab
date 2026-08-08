from geniie_lab.experiments.repetition_experiment import ExperimentRunner

from tests.conftest import make_settings
from tests.fakes import parse_jsonl


def test_last_stage_repeats_with_repetition_numbers(patched_services, capsys):
    settings = make_settings(name="test_repetition", plan=["query"], loop_num_per_topic=3)
    ExperimentRunner(settings=settings).run()
    records = parse_jsonl(capsys.readouterr().out)

    # plan[:-1] is empty; the single "query" stage repeats 3x per topic.
    assert [r["stage"] for r in records] == ["query"] * 6
    assert [r["repetition"] for r in records] == [1, 2, 3, 1, 2, 3]
    assert [r["topic_id"] for r in records] == ["t1"] * 3 + ["t2"] * 3


def test_each_repetition_starts_from_cloned_base_memory(patched_services, capsys):
    # Every repeated query-generation call must see the same (empty) base history:
    # the runner clones base memory before each repetition instead of letting
    # repetitions contaminate one another.
    settings = make_settings(plan=["query"], loop_num_per_topic=3)
    ExperimentRunner(settings=settings).run()
    capsys.readouterr()

    assert [call for call in patched_services.llm.calls] == [("Query", 0)] * 6


def test_prefix_stages_run_once_and_are_shared_by_repetitions(patched_services, capsys):
    # plan = query, ranking, click: the prefix (query, ranking) runs once per
    # topic; only "click" repeats. Each click call must start from the same
    # base history (the 2 messages added by the query stage).
    settings = make_settings(plan=["query", "ranking", "click"], loop_num_per_topic=2)
    ExperimentRunner(settings=settings).run()
    records = parse_jsonl(capsys.readouterr().out)

    per_topic_stages = ["query", "ranking", "click", "click"]
    assert [r["stage"] for r in records] == per_topic_stages * 2

    clicks = [r for r in records if r["stage"] == "click"]
    assert [r["repetition"] for r in clicks] == [1, 2, 1, 2]

    click_calls = [c for c in patched_services.llm.calls if c[0] == "Clicks"]
    # 2 topics x 2 repetitions, each starting from the cloned base history of
    # exactly 2 messages (user instruction + assistant response of the query stage).
    assert click_calls == [("Clicks", 2)] * 4
