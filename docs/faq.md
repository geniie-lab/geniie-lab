# FAQ

## What is a GII?
GII stands for Generative Information Intermediary, and is defined as a specialised application of LLMs designed to retrieve external resources in support of downstream tasks in retrieval augemend applications.

## Do I need to bring OpenSearch clients if I'm only interested in query formulation stage?
No, if you're only setting `query` in `plan=[]` in `Session Experiment` or `Repetition Experiment`, you don't need to bring tools such as opensearch clients.

## How can I use Pyterrier/Pyserini as tools?
We currently support only opensearch as search tools. We will look into other IR tools such as Pyterrier and Pyserini as a future plan.

## How can I run experiments in non-English settings
Currently, all instructions and communication with LLMs are written in English. You can manually change these instructions to run the experiment in non-English settings. Please look at [Advanced Topics](advanced/index.md) to find out what files to edit.

## How can I use a dataset that is not available via ir_datasets
If you would like to use a dataset that is not available in `ir_datasets`, you can write a data bridge class. See [Advanced Options](advanced/index.md) to learn how to do it using Amazon ESCI dataset as an example.

## How can I apply AI persona on GII?
One way to do this is to change `system_prompt` parameter in `ModelDescription` to reflect the characteristics of your AI persona. If you list mulitple `ModelDescription` instances with different `system_prompt` parameters in `models[]`, the experiment allows you to compare different AI persona for GII.

## How can I add a canonical document to input?
We do not currently support a canonical document as input. We will look into this as a future plan.

## Are there any tools to analyse the output data?
No, we decided not to include analytics tools as part of `geniie-lab` since how to analyse response data closely depends on diverse research questions formulated by researchers.