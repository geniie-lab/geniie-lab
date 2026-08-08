# Session experiment template for REASONING ("thinking") models.
#
# Reasoning models expose their internal deliberation as a trace next to the
# structured answer. geniie-lab captures that trace per stage (`thinking` /
# `thinking_token` in the session log) and offers two length controls. This
# script mirrors run_session_experiment.py with those parameters configured;
# instruct models don't need any of them.
#
# Verified stacks (all expose the trace as message.reasoning):
#   - gpt-oss-120b on vLLM (--reasoning-parser openai_gptoss) or Amazon Bedrock
#   - Qwen3-Next-80B-A3B-Thinking on vLLM (--reasoning-parser qwen3)
#   - NVIDIA-Nemotron-3-Super-120B on vLLM (--reasoning-parser nemotron_v3)

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
from geniie_lab.dataclasses.topic import FullTopic
from geniie_lab.dataclasses.setting import ExperimentSettings, StageConfig
from geniie_lab.experiments.session_experiment import ExperimentRunner

load_dotenv()

my_settings = ExperimentSettings(
    name="my_reasoning_session_experiment",
    task=TaskDescription(
        name="High-Recall Retrieval",
        description="Find as many different relevant documents as possible for a given search topic from a given document collection using a provided search tool.",
        measurement=[ir_measures.nDCG@10, ir_measures.MRR@10],
        start_offset=0,
        serp_size=10,
    ),
    topicset=TopicDescription(
        name="aquaint/trec-robust-2005",
        type="ir_datasets",
        topic_class=FullTopic
    ),
    corpus=CorpusDescription(
        name="Aquaint",
        description="A document collection of about 1M English newswire text. Sources include the Xinhua News Service (1996-2000), the New York Times News Service (1998-2000), and the Associated Press Worldstream News Service (1998-2000).",
        index_name="aquaint_bm25",
    ),
    models=[
        ModelDescription(
            type="vllm",
            name="openai/gpt-oss-120b",
            token_length=128000,  # set to your model's max input token length
            system_prompt="You're a helpful assistant",
            temperature=0.0,
            top_p=1.0,
            # gpt-oss plans its own reasoning length via a trained-in dial:
            # "low" | "medium" (provider default) | "high". Roughly 100 / 900 /
            # 2500 reasoning characters per call in our measurements. Omit for
            # the provider default.
            # reasoning_effort="low",
            # Hard server-side cap on reasoning tokens, as a runaway guard.
            # Enforced by vLLM's qwen3-class reasoning parsers; silently
            # ignored by providers without the feature (including gpt-oss).
            thinking_token_budget=1024,
        ),
        # ModelDescription(
        #     type="vllm",
        #     name="Qwen3-Next-80B-A3B-Thinking",
        #     token_length=131072,  # set to your model's max input token length
        #     system_prompt="You're a helpful assistant",
        #     temperature=0.0,
        #     top_p=1.0,
        #     # Qwen3-Thinking overthinks by design (thousands of tokens per
        #     # call); a budget is strongly recommended. For a graceful cut
        #     # instead of a mid-sentence one, start vLLM with the transition
        #     # phrase baked into the reasoning end string:
        #     #   --reasoning-parser qwen3
        #     #   --reasoning-config '{"reasoning_start_str": "<think>",
        #     #     "reasoning_end_str": "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"}'
        #     thinking_token_budget=1024,
        # ),
        # ModelDescription(
        #     type="bedrock",
        #     name="openai.gpt-oss-120b",
        #     token_length=128000,  # set to your model's max input token length
        #     system_prompt="You're a helpful assistant",
        #     temperature=0.0,
        #     top_p=1.0,
        #     # reasoning_effort="low",
        # ),
    ],
    tools=[
        ToolDescription(
            name="opensearch",
            ranking_model="bm25",
            index_name="aquaint_bm25",
            description="It allows you to perform searches using keywords only and employs the BM25 ranking model to order results.",
        ),
    ],
    stages={
        # The thinking trace TEXT is logged per stage, opt-in
        # (log_thinking=False by default). thinking_token — the trace's
        # estimated token count — is always recorded for every LLM stage, so
        # thinking cost stays measurable even where the text is suppressed.
        "query": StageConfig(
            instruction="""
                Review the provided descriptions of task, corpus, tool and search topic. Then, formulate a search query.
            """,
            log_thinking=True,
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
            log_thinking=True,
        ),
    },
    plan=["query", "ranking", "click", "relevance", "reformulate", "ranking"],
    topic_ids="1:10",
    full_log=False
)

if __name__ == "__main__":
    runner = ExperimentRunner(settings=my_settings)
    runner.run()
