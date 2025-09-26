# Agentic Experiment

In the agentic experiment, researchers can examine the GII response for the decision-making of search stages. For example, after the initial set of search stages, `{query, ranking, clicking, relevance judgement}`, the GII is asked for the next action such as query reformulation, next serp page, more clicks, or end the task. The experiment continue unitl either `max_actions` or GII's decision of task termination.

See [Common Settings](common_settings.md) and [Session Experiment](session_experiment.md) first to learn the overall experimental settings, while we do not use `plan` in this experiment.


## Next Action stage

Since we let GII to decide the next action, a new stage is added. The sequence of search stages is defined in `ExperimentRunner` class of `geniie-lab/experiments/agentic_experiemnt.py`. See [Advanced Topics](../advanced/index.md) to learn how to configure these sequences.

- `Submit_New_Query`: The next action will be a sequence of `query`, `ranking`, `click`, `relevance`, and `next_action`
- `Click_Document`: The next action will be a sequence of `click` (on the same SERP), `relevance`, and `next_action`
- `GO_NEXT_RESULT_PAGE`: The next action will be sequence of `ranking` (with `start_offset += serp_size`), `click`, `relevance`, and `next_action`

```
    stages={
        ...
        "next_action": StageConfig(
            instruction="""
                Based on your progress so far, decide on the next action to take toward completing the task.

                Choose one of the following actions and provide a brief explanation for your choice:
                - **Submit_New_Query**: Submit a new search query.
                - **Click_Document**: Select a set of documents from the current search results.
                - **GO_NEXT_RESULT_PAGE**: Navigate to the next page of search results for the current query.
                - **End_Task**: End the current task.
            """
        ),
    },
    max_actions=5,
    ...
```

Furthermore, we need to set `max_action` in the agentic experiment.

- `max_action`: The maximu number of actions to be taken before termination. Note that GII might choose `End_Task` option from the `next_action` stage before `max_action` reaches. Default: `None`

## Sample output

- `model`: `gllama3.3:70b-instruct-q4_K_M`
- `dataset`: `quaint/trec-robust-2005`
- `ranking_model`: `bm25`
- `max_actions`: `10`

```
{"session_name": "my_agentic_experiment", "model": "llama3.3:70b-instruct-q4_K_M", "task": "High-Recall Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "query": "Hubble Telescope AND (achievements OR discoveries OR breakthroughs)", "start": 0, "size": 10, "repetition": 1, "reason": null, "stage": "query", "created_at": "2025-07-02T02:31:13.317727+00:00"}
{"session_name": "my_agentic_experiment", "model": "llama3.3:70b-instruct-q4_K_M", "ranker": "bm25", "task": "High-Recall Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "doc_ids": ["XIE19961108.0181", ..., "NYT19990719.0430"], "start": 0, "size": 10, "performance": {"nDCG@10": 0.4922818839877925, "R@10": 0.06976744186046512}, "repetition": 1, "stage": "ranking", "created_at": "2025-07-02T02:31:13.430043+00:00"}
{"session_name": "my_agentic_experiment", "model": "llama3.3:70b-instruct-q4_K_M", "task": "High-Recall Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "rankings": [2, 3, 7, 10], "repetition": 1, "reason": null, "stage": "click", "created_at": "2025-07-02T02:31:20.637674+00:00"}
{"session_name": "my_agentic_experiment", "model": "llama3.3:70b-instruct-q4_K_M", "task": "High-Recall Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "docid": "APW19990108.0103", "label": "Relevance.RELEVANT", "qrel_label": 2, "repetition": 1, "stage": "rel_judge", "created_at": "2025-07-02T02:31:28.490257+00:00"}
...
{"session_name": "my_agentic_experiment", "model": "llama3.3:70b-instruct-q4_K_M", "task": "High-Recall Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "action": "GO_NEXT_RESULT_PAGE", "action_num": 5, "repetition": 1, "reason": "The current page of search results contains some relevant documents, but it's likely that there are more relevant documents in the subsequent pages. By navigating to the next page, we can potentially find additional relevant documents and increase the overall number of relevant documents found.", "stage": "next_action", "created_at": "2025-07-02T02:32:15.932589+00:00"}
{"session_name": "my_agentic_experiment", "model": "llama3.3:70b-instruct-q4_K_M", "ranker": "bm25", "task": "High-Recall Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "doc_ids": ["XIE19980104.0071", ..., "NYT19990310.0333"], "start": 10, "size": 10, "performance": {"nDCG@10": 0.2626633251188907, "R@10": 0.023255813953488372}, "repetition": 1, "stage": "ranking", "created_at": "2025-07-02T02:32:16.047583+00:00"}
...
```