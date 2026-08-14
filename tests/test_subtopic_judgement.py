"""Tests for per-subtopic relevance judgement (subtopic-presenting topics,
the SubtopicRelevanceJudgement schema, and the recorded output)."""

import pytest
from pydantic import ValidationError

from geniie_lab.dataclasses.output import SubtopicRelevanceJudgementExperimentOutput
from geniie_lab.dataclasses.setting import StageConfig
from geniie_lab.dataclasses.topic import (
    NtcirIntentSubtopicsTopic,
    Subtopic,
    TrecDiversitySubtopicsTopic,
)
from geniie_lab.response import (
    RUBRIC_ORDER,
    RubricRelevance,
    SubtopicRelevance,
    SubtopicRelevanceJudgement,
)


def make_trec_topic():
    return TrecDiversitySubtopicsTopic(
        id="20",
        title="defender",
        description="Find information about the Land Rover Defender.",
        topic_type="ambiguous",
        subtopics=[
            Subtopic(number="1", text="Land Rover Defender SUV", intent_type="inf"),
            Subtopic(number="2", text="Defender arcade game", intent_type="nav"),
        ],
    )


class TestSubtopicPresentingTopics:
    def test_trec_presents_all_information(self):
        rendered = str(make_trec_topic())
        assert "- **Title**: defender" in rendered
        assert "Land Rover Defender." in rendered          # description
        assert "ambiguous" in rendered                      # topic type
        assert "1. (inf) Land Rover Defender SUV" in rendered
        assert "2. (nav) Defender arcade game" in rendered

    def test_ntcir_presents_probabilities_and_types(self):
        topic = NtcirIntentSubtopicsTopic(
            id="0101",
            title="日露戦争",
            subtopics=[
                Subtopic(number="1", text="cause", probability=0.16, intent_type=None),
                Subtopic(number="2", text="timeline", probability=0.14, intent_type="inf"),
            ],
        )
        rendered = str(topic)
        assert "1. (p=0.160) cause" in rendered
        assert "2. (p=0.140, inf) timeline" in rendered

    def test_parent_classes_stay_title_only(self):
        # The presentation split: parents never show subtopics.
        from geniie_lab.dataclasses.topic import TrecDiversityTopic
        parent = TrecDiversityTopic(**vars(make_trec_topic()))
        assert str(parent) == "- **Title**: defender"


class TestRubricRelevance:
    def test_values_are_what_the_model_emits(self):
        assert RubricRelevance.NOT_ADDRESSED == "NotAddressed"
        assert RubricRelevance.COMPLETELY_SATISFIED == "CompletelySatisfied"

    def test_rubric_order_supports_at_least_comparisons(self):
        satisfied = RUBRIC_ORDER.index(RubricRelevance.NEED_SATISFIED)
        assert RUBRIC_ORDER.index(RubricRelevance.ON_SUBTOPIC_ONLY) < satisfied
        assert RUBRIC_ORDER.index(RubricRelevance.COMPLETELY_SATISFIED) > satisfied
        assert len(RUBRIC_ORDER) == len(RubricRelevance)


class TestSubtopicRelevanceJudgementSchema:
    def test_label_must_be_a_rubric_value(self):
        with pytest.raises(ValidationError):
            SubtopicRelevance(subtopic=1, label="VeryRelevant", evidence="x")
        with pytest.raises(ValidationError):
            SubtopicRelevance(subtopic=1, label=2, evidence="x")

    def test_evidence_defaults_empty(self):
        item = SubtopicRelevance(subtopic=1, label=RubricRelevance.NOT_ADDRESSED)
        assert item.evidence == ""

    def test_reason_required(self):
        with pytest.raises(ValidationError):
            SubtopicRelevanceJudgement(labels=[])


class TestOutputRecord:
    def test_round_trips_to_json(self):
        record = SubtopicRelevanceJudgementExperimentOutput(
            session_name="s", model="m", task="t", dataset="d",
            topic_id="20", docid="doc-1",
            labels=[{"subtopic": 1, "label": "NeedSatisfied", "evidence": "quote"}],
            qrel_labels={"1": 1, "2": 0},
            reason="because",
        )
        blob = record.to_json()
        assert '"labels"' in blob and '"qrel_labels"' in blob
        assert record.stage == "rel_judge"

    def test_record_carries_no_derived_document_label(self):
        # Per-subtopic labels are recorded; document relevance is analysis.
        fields = SubtopicRelevanceJudgementExperimentOutput.__dataclass_fields__
        assert "label" not in fields and "threshold" not in fields
        assert "qrel_label" not in fields

    def test_stageconfig_carries_response_model(self):
        config = StageConfig(response_model=SubtopicRelevanceJudgement)
        assert config.response_model is SubtopicRelevanceJudgement


class TestSubtopicQrelLabels:
    """Every presented subtopic gets a label; absent means nonrelevant (#57)."""

    @staticmethod
    def build(graded, presented):
        # mirrors the stage: graded rows from the qrels, one entry per subtopic
        return {str(s): graded.get(str(s), 0) for s in presented}

    def test_absent_subtopics_are_zero_not_missing(self):
        labels = self.build({"2": 1}, presented=[1, 2, 3])
        assert labels == {"1": 0, "2": 1, "3": 0}

    def test_document_relevant_to_nothing_is_all_zero(self):
        labels = self.build({"0": 0}, presented=[1, 2])
        assert labels == {"1": 0, "2": 0}

    def test_every_presented_subtopic_is_covered(self):
        presented = [1, 2, 3, 4, 5]
        assert set(self.build({"3": 2}, presented)) == {"1", "2", "3", "4", "5"}
