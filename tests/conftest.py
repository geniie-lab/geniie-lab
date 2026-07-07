from types import SimpleNamespace

import ir_datasets
import ir_measures
import pytest

from geniie_lab.dataclasses.description import (
    CorpusDescription,
    ModelDescription,
    TaskDescription,
    ToolDescription,
    TopicDescription,
)
from geniie_lab.dataclasses.setting import ExperimentSettings
from geniie_lab.dataclasses.topic import FullTopic
from geniie_lab.services.llm.llm_service_factory import LLMServiceFactory
from geniie_lab.services.opensearch.opensearch_client_factory import OpenSearchClientFactory

from tests.fakes import FakeLLMService, FakeOpenSearchClient, default_dataset


@pytest.fixture
def fake_llm():
    return FakeLLMService()


@pytest.fixture
def fake_search():
    return FakeOpenSearchClient()


@pytest.fixture
def fake_dataset():
    return default_dataset()


@pytest.fixture
def patched_services(monkeypatch, fake_llm, fake_search, fake_dataset):
    """Route the runners' three external seams to fakes: ir_datasets and the
    two service factories. No network, no OpenSearch, no LLM API."""
    monkeypatch.setattr(ir_datasets, "load", lambda name: fake_dataset)
    monkeypatch.setattr(
        LLMServiceFactory, "create_llm_service",
        lambda self, genai_type: fake_llm,
    )
    monkeypatch.setattr(
        OpenSearchClientFactory, "create_opensearch_client",
        lambda self, settings, tool: fake_search,
    )
    return SimpleNamespace(llm=fake_llm, search=fake_search, dataset=fake_dataset)


def make_settings(name="test_experiment", plan=None, loop_num_per_topic=1,
                  max_actions=None) -> ExperimentSettings:
    return ExperimentSettings(
        name=name,
        task=TaskDescription(
            name="High-Recall Retrieval",
            description="Find as many relevant documents as possible.",
            measurement=[ir_measures.nDCG@10, ir_measures.MRR@10],
            start_offset=0,
            serp_size=10,
        ),
        topicset=TopicDescription(
            name="fake/dataset",
            type="ir_datasets",
            topic_class=FullTopic,
        ),
        corpus=CorpusDescription(
            name="Fake Corpus",
            description="A fake corpus for testing.",
            index_name="fake_bm25",
        ),
        models=[
            ModelDescription(type="fake", name="fake-model", token_length=100000),
        ],
        tools=[
            ToolDescription(
                name="opensearch",
                ranking_model="bm25",
                index_name="fake_bm25",
                description="A fake search tool.",
            ),
        ],
        plan=plan,
        loop_num_per_topic=loop_num_per_topic,
        max_actions=max_actions,
    )
