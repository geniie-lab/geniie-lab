# Third-party libraries
from dotenv import load_dotenv
import ir_measures
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local application imports
from geniie_lab.dataclasses.description import (
    CorpusDescription,
    ModelDescription,
    TaskDescription,
    ToolDescription,
    TopicDescription,
)
from geniie_lab.dataclasses.topic import (
    TitleDescriptionNarrativeTopic, FullTopic,
    TitleDescriptionTopic,
    TitleNarrativeTopic,
    TitleOnlyTopic
)
from geniie_lab.dataclasses.setting import ExperimentSettings, StageConfig
from geniie_lab.experiments.session_experiment import ExperimentRunner
import geniie_lab.databridges.amazon_esci.amazon_esci

load_dotenv()

my_settings = ExperimentSettings(
    name="my_session_experiment",
    task=TaskDescription(
        name = "High-Precision Retrieval",
        description = "Find the most relevant documents at the top rank for a given search topic from a given document collection using a provided search tool.",
        measurement=[ir_measures.nDCG@10, ir_measures.MRR@10],
        start_offset=0,
        serp_size=10,
    ),
    topicset=TopicDescription(
        name="custom-amazon-esci/train",
        type="ir_datasets",
        topic_class=TitleOnlyTopic
    ),
    corpus=CorpusDescription(
        name="Amazon ESCI",
        description="A large dataset of difficult search queries, released with the aim of fostering research in the area of semantic matching of queries and products.",
        index_name="esci_bm25",
    ),
    models=[
        ModelDescription(
            type="openai",
            name="gpt-4.1-mini-2025-04-14",
            system_prompt="You're a helpful assistant",
            temperature=0.0,
        ),
    ],
    tools=[
        # This is a dummy data
        ToolDescription(
            name="opensearch",
            ranking_model="bm25",
            index_name="esci_bm25",
            host="localhost",
            port=9200,
            description="It allows you to perform searches using keywords only and employs the BM25 ranking model to order results.",
        ),
    ],
    stages={
        "query": StageConfig(
            instruction="""
                Review the provided descriptions of task, corpus, tool and search topic. Then, formulate a search query.
            """,
        ),
        "ranking": StageConfig(
            instruction=""
        ),
        "click": StageConfig(
            instruction="""
                Select a set of documents that are likely to contain relevant information to the search topic. Return an empty list if none of the results appears relevant.
            """,
        ),
        "relevance": StageConfig(
            instruction="""
                Evaluate the relevance of the document based on the search topic description and its narrative.
            """,
        ),
        "reformulate": StageConfig(
            instruction="""
                Formulate another search query to find new relevant documents.
            """,
        ),
    },
    plan=["query"],
    max_topics=5,
    full_log=False
)

if __name__ == "__main__":
    runner = ExperimentRunner(settings=my_settings)
    runner.run()
