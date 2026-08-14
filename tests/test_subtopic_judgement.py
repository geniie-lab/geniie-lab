"""Tests for per-subtopic relevance judgement (subtopic-presenting topics,
SubtopicRelevanceJudgement schema, and the derived binary label)."""

import pytest
from pydantic import ValidationError

from geniie_lab.dataclasses.output import SubtopicRelevanceJudgementExperimentOutput
from geniie_lab.dataclasses.setting import StageConfig
from geniie_lab.dataclasses.topic import (
    NtcirIntentSubtopicsTopic,
    Subtopic,
    TrecDiversitySubtopicsTopic,
)
from geniie_lab.response import SubtopicGrade, SubtopicRelevanceJudgement


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


class TestSubtopicRelevanceJudgementSchema:
    def test_grade_bounds_enforced(self):
        with pytest.raises(ValidationError):
            SubtopicGrade(subtopic=1, grade=4, evidence="x")
        with pytest.raises(ValidationError):
            SubtopicGrade(subtopic=1, grade=-1, evidence="x")

    def test_evidence_defaults_empty(self):
        assert SubtopicGrade(subtopic=1, grade=0).evidence == ""

    def test_reason_required(self):
        with pytest.raises(ValidationError):
            SubtopicRelevanceJudgement(assessments=[])


class TestDerivedLabel:
    @staticmethod
    def derive(grades, threshold=2):
        judgement = SubtopicRelevanceJudgement(
            assessments=[SubtopicGrade(subtopic=i + 1, grade=g, evidence="e" if g else "")
                         for i, g in enumerate(grades)],
            reason="r",
        )
        return any(a.grade >= threshold for a in judgement.assessments)

    def test_all_zeros_is_not_relevant(self):
        assert self.derive([0, 0, 0]) is False

    def test_grade_one_is_not_relevant(self):
        assert self.derive([1, 1, 0]) is False

    def test_any_grade_two_is_relevant(self):
        assert self.derive([0, 2, 0]) is True

    def test_threshold_is_configurable(self):
        assert self.derive([1, 1, 1], threshold=1) is True
        assert self.derive([2, 2, 2], threshold=3) is False


class TestOutputRecord:
    def test_round_trips_to_json(self):
        record = SubtopicRelevanceJudgementExperimentOutput(
            session_name="s", model="m", task="t", dataset="d",
            topic_id="20", docid="doc-1", label="Relevance.RELEVANT",
            assessments=[{"subtopic": 1, "grade": 2, "evidence": "quote"}],
            subtopic_qrel_labels={"1": 1, "2": 0},
            qrel_label=1, threshold=2, reason="because",
        )
        blob = record.to_json()
        assert '"assessments"' in blob and '"subtopic_qrel_labels"' in blob
        assert record.stage == "rel_judge"

    def test_stageconfig_carries_model_and_threshold(self):
        config = StageConfig(response_model=SubtopicRelevanceJudgement,
                             relevance_threshold=2)
        assert config.response_model is SubtopicRelevanceJudgement
        assert config.relevance_threshold == 2


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
