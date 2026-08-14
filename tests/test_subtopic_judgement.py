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
    GradedRelevance,
    Relevance,
    RubricRelevance,
    SubtopicRelevance,
    SubtopicRelevanceJudgement,
)

RubricJudgement = SubtopicRelevanceJudgement[RubricRelevance]
GradedJudgement = SubtopicRelevanceJudgement[GradedRelevance]


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


class TestLabelScales:
    def test_values_are_what_the_model_emits(self):
        assert RubricRelevance.NOT_ADDRESSED == "NotAddressed"
        assert RubricRelevance.COMPLETELY_SATISFIED == "CompletelySatisfied"
        assert GradedRelevance.PARTIALLY_RELEVANT == "PartiallyRelevant"
        assert GradedRelevance.HIGHLY_RELEVANT == "HighlyRelevant"

    def test_rank_supports_at_least_comparisons(self):
        assert RubricRelevance.ON_SUBTOPIC_ONLY.rank < RubricRelevance.NEED_SATISFIED.rank
        assert RubricRelevance.COMPLETELY_SATISFIED.rank > RubricRelevance.NEED_SATISFIED.rank
        assert GradedRelevance.NOT_RELEVANT.rank == 0
        assert GradedRelevance.HIGHLY_RELEVANT.rank == len(GradedRelevance) - 1


class TestSubtopicRelevanceJudgementSchema:
    def test_label_must_belong_to_the_chosen_scale(self):
        with pytest.raises(ValidationError):
            SubtopicRelevance[RubricRelevance](subtopic=1, label="VeryRelevant", evidence="x")
        with pytest.raises(ValidationError):
            SubtopicRelevance[RubricRelevance](subtopic=1, label=2, evidence="x")

    def test_scales_do_not_accept_each_others_values(self):
        with pytest.raises(ValidationError):
            GradedJudgement(labels=[{"subtopic": 1, "label": "NeedSatisfied"}], reason="r")
        with pytest.raises(ValidationError):
            RubricJudgement(labels=[{"subtopic": 1, "label": "HighlyRelevant"}], reason="r")

    def test_binary_relevance_is_a_usable_scale(self):
        # Relevance predates the ordered scales and stays usable as a label.
        judgement = SubtopicRelevanceJudgement[Relevance](
            labels=[{"subtopic": 1, "label": "Relevant", "evidence": "q"}], reason="r")
        assert judgement.labels[0].label is Relevance.RELEVANT

    def test_evidence_defaults_empty(self):
        item = SubtopicRelevance[RubricRelevance](
            subtopic=1, label=RubricRelevance.NOT_ADDRESSED)
        assert item.evidence == ""

    def test_reason_required(self):
        with pytest.raises(ValidationError):
            RubricJudgement(labels=[])

    def test_stage_dispatch_sees_the_generic_base(self):
        # session_experiment branches on isinstance against the unparametrised
        # base; parametrised instances must still match.
        judgement = RubricJudgement(
            labels=[{"subtopic": 1, "label": "NeedSatisfied", "evidence": "q"}], reason="r")
        assert isinstance(judgement, SubtopicRelevanceJudgement)


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
        config = StageConfig(response_model=RubricJudgement)
        assert config.response_model is RubricJudgement


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
