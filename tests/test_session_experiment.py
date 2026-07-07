import math

from geniie_lab.experiments.session_experiment import ExperimentRunner

from tests.conftest import make_settings
from tests.fakes import FAKE_TOTAL_TOKEN, parse_jsonl


def test_full_plan_emits_expected_record_sequence(patched_services, capsys):
    settings = make_settings(
        name="test_session",
        plan=["query", "ranking", "click", "relevance", "reformulate"],
    )
    ExperimentRunner(settings=settings).run()
    records = parse_jsonl(capsys.readouterr().out)

    # 2 topics x 5 stages, in plan order.
    expected_stages = ["query", "ranking", "click", "rel_judge", "reformulation"]
    assert [r["stage"] for r in records] == expected_stages * 2
    assert [r["topic_id"] for r in records] == ["t1"] * 5 + ["t2"] * 5

    query_rec = records[0]
    assert query_rec["query"] == "first query"
    assert query_rec["session_name"] == "test_session"
    assert query_rec["model"] == "fake-model"
    assert query_rec["dataset"] == "fake/dataset"
    assert query_rec["total_token"] == FAKE_TOTAL_TOKEN

    reform_rec = records[4]
    assert reform_rec["query"] == "reformulated query"


def test_ranking_evaluates_serp_in_display_order(patched_services, capsys):
    # B1: the relevant doc d1 is at rank 1 (highest retrieval score), so the
    # metrics must reflect a top-ranked hit -- not the reversed SERP.
    settings = make_settings(plan=["query", "ranking"])
    ExperimentRunner(settings=settings).run()
    records = parse_jsonl(capsys.readouterr().out)

    rank_t1 = next(r for r in records if r["stage"] == "ranking" and r["topic_id"] == "t1")
    assert rank_t1["doc_ids"] == ["d1", "d2"]
    assert rank_t1["performance"]["nDCG@10"] == 1.0
    assert rank_t1["performance"]["RR@10"] == 1.0

    # t2 has no qrels at all: ir_measures aggregates to NaN, which is emitted
    # verbatim into the JSONL (a bare NaN token -- tolerated by Python's json
    # module but invalid strict JSON; recorded here as current behavior).
    rank_t2 = next(r for r in records if r["stage"] == "ranking" and r["topic_id"] == "t2")
    assert math.isnan(rank_t2["performance"]["RR@10"])


def test_relevance_stage_reports_qrel_label(patched_services, capsys):
    settings = make_settings(plan=["query", "ranking", "click", "relevance"])
    ExperimentRunner(settings=settings).run()
    records = parse_jsonl(capsys.readouterr().out)

    rel_t1 = next(r for r in records if r["stage"] == "rel_judge" and r["topic_id"] == "t1")
    assert rel_t1["docid"] == "d1"
    assert rel_t1["qrel_label"] == 1
    rel_t2 = next(r for r in records if r["stage"] == "rel_judge" and r["topic_id"] == "t2")
    assert rel_t2["qrel_label"] == 0


def test_stage_error_stops_plan_for_topic(patched_services, capsys):
    # B2: "click" errors without a SERP; the remaining stages of the plan must
    # not run for that topic.
    settings = make_settings(plan=["click", "query"])
    ExperimentRunner(settings=settings).run()
    records = parse_jsonl(capsys.readouterr().out)

    assert records == []  # click errored; query never ran for either topic
    assert patched_services.llm.calls == []
