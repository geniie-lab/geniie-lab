# geniie-lab

Welcome to the geniie-lab documentation!

`geniie-lab` is an open-source testbed that allows you to instruct large language models to perform search actions such as querying, ranking, clicking, judging relevance, and reformulating across configurable, modular stages. Researchers can configure an experiment using many [parameters](parameters.md) to investigate the impact of instruction factors on their search responses.

![](gii.png)

## Types of experiment

- [Session Experiment](experiments/session_experiment.md): researchers can examine the GII response of four search stages such as querying, clicking, relevance judgement, and query reformulation.
- [Repetition Experiment](experiments/repetition_experiment.md): researchers can examine the consistency (replicability) of GII responses at a stage in a session.
- [Agentic Experiment](experiments/agentic_experiment.md): researchers can examine the GII response for the decision-making of search stages.


## Prerequisites

You need to bring these items to use geniie-lab.

1. A test collection accessible via `ir_datasets`
1. LLMs (OpenAI, Gemini, or local models via `ollama`)
1. Opensearch client that can search the corpus of the test collection
    - Note that an opensearch client is not needed if you're only interested in `query` stage in the experiment.

## Getting Started

See [Getting Started](getting_started/index.md) to learn how to install and set up `geniie-lab`

## Advanced Topics

See [Advanced Topics](advanced/index.md) to learn how to configure `geniie-lab` in depth to match your needs.


## FAQ

See [FAQ](faq.md) for frequently asked questions regarding `geniie-lab`

