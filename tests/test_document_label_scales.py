"""The document-level relevance label scale is selectable per experiment,
in all three runners, and defaults to the binary Relevance scale."""
from geniie_lab.dataclasses.setting import StageConfig
from geniie_lab.experiments.agentic_experiment import ExperimentRunner as AgenticRunner
from geniie_lab.experiments.repetition_experiment import ExperimentRunner as RepetitionRunner
from geniie_lab.experiments.session_experiment import ExperimentRunner as SessionRunner
from geniie_lab.response import GradedRelevance, RelevanceJudgement

from tests.conftest import make_settings
from tests.fakes import parse_jsonl

GRADED = StageConfig(response_model=RelevanceJudgement[GradedRelevance])


def rel_labels(capsys):
    return [r["label"] for r in parse_jsonl(capsys.readouterr().out)
            if r["stage"] == "rel_judge"]


def test_session_defaults_to_the_binary_scale(patched_services, capsys):
    settings = make_settings(plan=["query", "ranking", "click", "relevance"])
    SessionRunner(settings=settings).run()

    assert set(rel_labels(capsys)) == {"Relevant"}


def test_session_honours_a_graded_scale(patched_services, capsys):
    settings = make_settings(plan=["query", "ranking", "click", "relevance"],
                             stages={"relevance": GRADED})
    SessionRunner(settings=settings).run()

    assert set(rel_labels(capsys)) == {"HighlyRelevant"}


def test_agentic_honours_a_graded_scale(patched_services, capsys):
    settings = make_settings(max_actions=10, stages={"relevance": GRADED})
    AgenticRunner(settings=settings).run()

    assert set(rel_labels(capsys)) == {"HighlyRelevant"}


def test_repetition_honours_a_graded_scale(patched_services, capsys):
    settings = make_settings(plan=["query", "ranking", "click", "relevance"],
                             loop_num_per_topic=2, stages={"relevance": GRADED})
    RepetitionRunner(settings=settings).run()

    assert set(rel_labels(capsys)) == {"HighlyRelevant"}


def test_the_model_asked_for_carries_the_chosen_scale(patched_services, capsys):
    settings = make_settings(plan=["query", "ranking", "click", "relevance"],
                             stages={"relevance": GRADED})
    SessionRunner(settings=settings).run()
    capsys.readouterr()

    asked = [name for name, _ in patched_services.llm.calls
             if name.startswith("RelevanceJudgement")]
    assert asked == ["RelevanceJudgement[GradedRelevance]"] * 2  # one per topic
