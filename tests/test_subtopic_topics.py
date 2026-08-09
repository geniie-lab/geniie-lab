"""Tests for the subtopic-bearing topic classes (issue #51).

Raw query objects are synthetic namedtuples mirroring the ir_datasets shapes
(TREC Web Track diversity; NTCIR INTENT registrations) -- no collection data.
"""

from collections import namedtuple

from geniie_lab.dataclasses.topic import (
    NtcirIntentTopic,
    Subtopic,
    SubtopicTopic,
    TrecDiversityTopic,
)

TrecSubtopic = namedtuple("TrecSubtopic", ["number", "text", "type"])
TrecWebTrackQuery = namedtuple(
    "TrecWebTrackQuery", ["query_id", "query", "description", "type", "subtopics"]
)
IntentSubtopic = namedtuple("IntentSubtopic", ["number", "probability", "description"])
Intent2Subtopic = namedtuple(
    "Intent2Subtopic", ["number", "probability", "intent_type", "description"]
)
IntentQuery = namedtuple("IntentQuery", ["query_id", "query", "subtopics"])


def make_trec_raw():
    return TrecWebTrackQuery(
        query_id="20",
        query="defender",
        description="Find information about the Land Rover Defender sport-utility vehicle.",
        type="ambiguous",
        subtopics=(
            TrecSubtopic(number="1", text="Land Rover Defender SUV", type="inf"),
            TrecSubtopic(number="2", text="Defender arcade game", type="nav"),
        ),
    )


def make_intent_raw():
    return IntentQuery(
        query_id="0101",
        query="日露戦争",  # a Japanese query string
        subtopics=(
            IntentSubtopic(number="1", probability=0.16, description="cause"),
            Intent2Subtopic(number="2", probability=0.14, intent_type="inf", description="timeline"),
        ),
    )


class TestTrecDiversityTopic:
    def test_field_mapping(self):
        topic = TrecDiversityTopic.from_ir_datasets(make_trec_raw())
        assert topic.id == "20"
        assert topic.title == "defender"
        assert topic.topic_type == "ambiguous"
        assert isinstance(topic, SubtopicTopic)
        assert [s.number for s in topic.subtopics] == ["1", "2"]
        assert topic.subtopics[0].text == "Land Rover Defender SUV"
        assert topic.subtopics[0].probability is None

    def test_rendering_is_query_only_per_track_protocol(self):
        # TREC 2009 overview, sec. 2: participants received only the query
        # field; description and subtopics must never reach the prompt.
        topic = TrecDiversityTopic.from_ir_datasets(make_trec_raw())
        rendered = str(topic)
        assert rendered == "- **Title**: defender"
        assert "sport-utility" not in rendered
        for subtopic in topic.subtopics:
            assert subtopic.text not in rendered

    def test_description_kept_as_data(self):
        topic = TrecDiversityTopic.from_ir_datasets(make_trec_raw())
        assert topic.description.startswith("Find information")

    def test_whitespace_normalised(self):
        raw = make_trec_raw()._replace(query="  defender \n ")
        topic = TrecDiversityTopic.from_ir_datasets(raw)
        assert topic.title == "defender"


class TestNtcirIntentTopic:
    def test_field_mapping(self):
        topic = NtcirIntentTopic.from_ir_datasets(make_intent_raw())
        assert topic.id == "0101"
        assert topic.title == "日露戦争"
        assert isinstance(topic, SubtopicTopic)
        assert topic.subtopics[0] == Subtopic(
            number="1", text="cause", probability=0.16, intent_type=None
        )
        assert topic.subtopics[1].intent_type == "inf"

    def test_rendering_is_query_only(self):
        topic = NtcirIntentTopic.from_ir_datasets(make_intent_raw())
        assert str(topic) == "- **Title**: 日露戦争"
        for subtopic in topic.subtopics:
            assert subtopic.text not in str(topic)
