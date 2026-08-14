# Session experiment with per-subtopic relevance judgement.
#
# The searcher is shown the numbered subtopics of the topic, and labels every
# one of them for each document it reads. Two settings make that happen:
#   - a subtopic-presenting topic class, so the model actually sees them;
#   - StageConfig(response_model=SubtopicRelevanceJudgement[scale]) on the
#     relevance stage, whose instruction defines what the labels mean.
# See docs/advanced/subtopic_judgement.md.

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
from geniie_lab.dataclasses.topic import TrecDiversitySubtopicsTopic
from geniie_lab.dataclasses.setting import ExperimentSettings, StageConfig
from geniie_lab.experiments.session_experiment import ExperimentRunner
from geniie_lab.response import RubricRelevance, SubtopicRelevanceJudgement

load_dotenv()

# The anchors that give the RubricRelevance labels their meaning. Swap the
# scale and this text together: the schema constrains what the model may emit,
# the instruction says what each label means, and nothing checks they agree.
#
# The last sentence is collection-specific — it restates what NIST told its
# TREC 2009 assessors. Another collection needs whatever its own track told
# its assessors, and another language in place of English.
#
# Note there is no sentence about quoting evidence: that requirement lives in
# the `evidence` field description, and duplicating it here is not needed.
RUBRIC_INSTRUCTION = """
    Evaluate the relevance of the document based on each subtopic of the search topic, one at a time and independently.
    For every subtopic, assign a label:
    NotAddressed - the document does not address this subtopic.
    OnSubtopicOnly - related to this subtopic, but does not satisfy the need it expresses.
    NeedSatisfied - satisfies the need expressed by this subtopic.
    CompletelySatisfied - is dedicated to this subtopic and satisfies it completely.
    Label NotAddressed for every subtopic if the document is not in English, or if its content is misleading or malicious.
"""

my_settings = ExperimentSettings(
    # The output does not record which label scale ran, so name it here: the
    # experiment name is logged as session_name on every row.
    name="my_subtopic_experiment_rubric",
    task=TaskDescription(
        name="High-Diversity Retrieval",
        description="Find as many different relevant documents as possible for each of the listed subtopics of the search topic from a given document collection using a provided search tool.",
        measurement=[ir_measures.alpha_nDCG@20],
        start_offset=0,
        serp_size=20,
    ),
    topicset=TopicDescription(
        # A subtopic-presenting topic class is required: the parent classes
        # render the query alone, so the model would be asked to label
        # subtopics it was never shown.
        name="<dataset whose queries carry subtopics>",
        type="ir_datasets",
        topic_class=TrecDiversitySubtopicsTopic,
    ),
    corpus=CorpusDescription(
        name="<corpus name>",
        description="<corpus description>",
        index_name="<index name>",
    ),
    models=[
        ModelDescription(
            type="openai",
            name="gpt-4.1-mini-2025-04-14",
            token_length=1000000,  # set to your model's max input token length
            system_prompt="You're a helpful assistant",
            temperature=0.0,
            top_p=1.0,
        ),
    ],
    tools=[
        ToolDescription(
            name="opensearch",
            ranking_model="bm25",
            index_name="<index name>",
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
            # The visited sentence assumes mark_visited_results is on (it is,
            # by default) — drop it when marking is off, since it would
            # describe something that never appears. The ranking-numbers
            # sentence is a crash guard: without it a model may click rank 0
            # and the runner abandons the topic.
            instruction="""
                Select a set of documents that are likely to contain relevant information to any subtopics of the search topic. Results you have already opened earlier in this session are marked visited=True. Refer to documents by the ranking numbers shown in the results, which start at 1. Return an empty list if none of the results appears relevant.
            """,
        ),
        "relevance": StageConfig(
            # Swapping to GradedRelevance means writing the anchors for its
            # four labels here; the label names alone do not define them.
            response_model=SubtopicRelevanceJudgement[RubricRelevance],
            instruction=RUBRIC_INSTRUCTION,
        ),
        "reformulate": StageConfig(
            instruction="""
                Formulate another search query to find new relevant documents to any subtopics of the search topic.
            """,
        ),
    },
    plan=["query", "ranking", "click", "relevance", "reformulate", "ranking"],
    max_topics=1,
    full_log=False
)

if __name__ == "__main__":
    runner = ExperimentRunner(settings=my_settings)
    runner.run()
