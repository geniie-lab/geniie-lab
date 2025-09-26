# Repetition Experiment

In the repetition experiment, researchers can examine the consistency (replicability) of GII responses at a stage in a session. For example, when a session is configured as `plan=["query"]`, then, the query formulation will be executed $N$ times, while when a session is `plan=["query", "ranking", "click"]`, the clicking stage will be repeated $N$ times. All the previous stages are executed only once.

See [Common Settings](common_settings.md) and [Session Experiment](session_experiment.md) first to learn the overall experimental settings and how to use `plan`.

## How to set a stage to repeat

In a repetition experiment, the last stage in the plan will be repeated for `loop_num_per_topic` times. All the previous items will be executed only once.

- `loop_num_per_topic`: Number of repetition for the last stage

If you're interested in repeating `query` stage:
```
    plan=["query"],
    loop_num_per_topic=2
```

If you're interested in repeating `click` stage:
```
    plan=["query", "ranking", "click"],
    loop_num_per_topic=2
```

If you're interested in repeating `relevance` stage:
```
    plan=["query", "ranking", "click", "relevance"],
    loop_num_per_topic=2,
```

Finally, if you're interested in repeating `reformulate` stage:
```
    plan=["query", "ranking", "click", "relevance", "reformulate"],
    loop_num_per_topic=2,
```

## Sample outputs

- `model`: `gemini-2.0-flash-lite-001`
- `dataset`: `quaint/trec-robust-2005`
- `ranking_model`: `bm25`
- `loop_num_per_topic`: `5`

### `plan=["query"]`

```
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "query": "Hubble Telescope AND (achievements OR accomplishments OR \"new data\" OR \"better quality data\" OR \"increased knowledge\" OR \"disproving theories\") NOT (repairs OR modifications OR shortcomings)", "start": 0, "size": 10, "repetition": 1, "reason": null, "stage": "query", "created_at": "2025-07-02T04:22:52.077303+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "query": "Hubble Telescope AND (achievements OR accomplishments OR \"new data\" OR \"better quality data\" OR \"increased knowledge\" OR \"disproving theories\") NOT (repairs OR modifications OR shortcomings)", "start": 0, "size": 10, "repetition": 2, "reason": null, "stage": "query", "created_at": "2025-07-02T04:22:54.573465+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "query": "Hubble Telescope AND (achievements OR accomplishments OR \"new data\" OR \"better quality data\" OR \"increased knowledge\" OR \"disproving theories\") NOT (repairs OR modifications OR shortcomings)", "start": 0, "size": 10, "repetition": 3, "reason": null, "stage": "query", "created_at": "2025-07-02T04:22:59.915404+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "query": "Hubble Telescope AND (achievements OR accomplishments OR \"new data\" OR \"better quality data\" OR \"increased human knowledge\" OR \"disproving theories\") NOT (repairs OR modifications OR shortcomings)", "start": 0, "size": 10, "repetition": 4, "reason": null, "stage": "query", "created_at": "2025-07-02T04:23:05.141619+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "query": "Hubble Telescope AND (achievements OR accomplishments OR \"new data\" OR \"better quality data\" OR \"increased human knowledge\" OR \"disproving theories\") NOT (repairs OR modifications OR shortcomings)", "start": 0, "size": 10, "repetition": 5, "reason": null, "stage": "query", "created_at": "2025-07-02T04:23:07.390999+00:00"}
```

###  `plan=["query", "ranking", "click"]`

```
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "query": "Hubble Telescope achievements data knowledge universe", "start": 0, "size": 10, "repetition": 1, "reason": null, "stage": "query", "created_at": "2025-07-02T04:27:37.739570+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "ranker": "bm25", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "doc_ids": ["APW19990108.0103", ..., "NYT19990310.0503"], "start": 0, "size": 10, "performance": {"nDCG@10": 0.4735717030454861, "RR@10": 0.5}, "repetition": 1, "stage": "ranking", "created_at": "2025-07-02T04:27:37.903353+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "rankings": [1, 2, 7, 9], "repetition": 1, "reason": null, "stage": "click", "created_at": "2025-07-02T04:27:43.512527+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "rankings": [1, 2, 7, 9], "repetition": 2, "reason": null, "stage": "click", "created_at": "2025-07-02T04:27:49.290060+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "rankings": [1, 2, 7, 9], "repetition": 3, "reason": null, "stage": "click", "created_at": "2025-07-02T04:27:52.098887+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "rankings": [1, 2, 7, 9], "repetition": 4, "reason": null, "stage": "click", "created_at": "2025-07-02T04:27:57.934037+00:00"}
{"session_name": "my_repetition_experiment", "model": "gemini-2.0-flash-lite-001", "task": "High-Precision Retrieval", "dataset": "aquaint/trec-robust-2005", "topic_id": "303", "rankings": [1, 2, 7, 9], "repetition": 5, "reason": null, "stage": "click", "created_at": "2025-07-02T04:28:00.674485+00:00"}
```