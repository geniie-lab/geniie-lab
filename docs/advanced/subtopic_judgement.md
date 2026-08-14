# Per-subtopic relevance judgement

By default the relevance stage asks the LLM for one label per document. For diversity and intent collections, where a topic is a set of subtopics, you can instead ask for one label *per subtopic*: the LLM reads the document once and labels it against every subtopic in a single call.

Two settings are needed, and they must agree with each other.

## 1. Present the subtopics to the LLM

The topic class controls what the LLM sees. The parent classes render the query alone, matching the original track protocols in which participants never saw the subtopics:

| Topic class | What the LLM sees |
|---|---|
| `TrecDiversityTopic` | query only |
| `TrecDiversitySubtopicsTopic` | title, description, topic type, numbered subtopics |
| `NtcirIntentTopic` | query only |
| `NtcirIntentSubtopicsTopic` | query, numbered intents with probabilities and types |

Use one of the `...SubtopicsTopic` classes. Pairing a parent class with a per-subtopic response model asks the LLM to label subtopics it was never shown.

```python
    topicset=TopicDescription(
        name="<dataset whose queries carry subtopics>",
        type="ir_datasets",
        topic_class=TrecDiversitySubtopicsTopic,
    ),
```

## 2. Choose the label scale

Set `response_model` on the `relevance` stage to `SubtopicRelevanceJudgement[scale]`, where the scale is an enum of the labels the LLM may emit:

| Scale | Labels |
|---|---|
| `RubricRelevance` | `NotAddressed`, `OnSubtopicOnly`, `NeedSatisfied`, `CompletelySatisfied` |
| `GradedRelevance` | `NotRelevant`, `PartiallyRelevant`, `Relevant`, `HighlyRelevant` |
| `Relevance` | `Relevant`, `NotRelevant` |

The two four-point scales are not interchangeable, and the choice changes what the LLM reports. `GradedRelevance` is a graded relevance scale: it asks how relevant the document is to the subtopic. `RubricRelevance` is a rubric over what happened to the information need the subtopic expresses — whether the document merely touches the subject or actually satisfies the need. A document can be highly relevant to a subtopic while satisfying none of it.

```python
from geniie_lab.response import RubricRelevance, SubtopicRelevanceJudgement

    "relevance": StageConfig(
        response_model=SubtopicRelevanceJudgement[RubricRelevance],
        instruction="""
            Judge the document against every subtopic listed in the search topic,
            in the order listed, using exactly one of these labels:
            - NotAddressed: the document does not address this subtopic.
            - OnSubtopicOnly: related to this subtopic but does not satisfy the need it expresses.
            - NeedSatisfied: satisfies the need expressed by this subtopic.
            - CompletelySatisfied: dedicated to this subtopic and satisfies it completely.
        """,
    ),
```

The scale must be given explicitly: a bare `SubtopicRelevanceJudgement` raises `specify a label scale, e.g. SubtopicRelevanceJudgement[RubricRelevance]`. Omit `response_model` altogether and the stage uses its own `RelevanceJudgement` schema, a single `Relevant`/`NotRelevant` label for the whole document.

The anchor wording belongs in the instruction, as above: the enum values alone name the levels but do not define them.

**The schema and the instruction are not cross-checked.** The scale constrains what the LLM may emit; the instruction says what each label means. If you change one, change the other.

`RubricRelevance` and `GradedRelevance` are ordered scales: each label has a `.rank` giving its position, so analysis code can ask for "at least `NeedSatisfied`" without hard-coding the list.

```python
[label for label in labels if label.rank >= RubricRelevance.NEED_SATISFIED.rank]
```

To add a scale of your own, subclass `OrderedLabels` in `geniie_lab/response.py` and declare the members from least to most relevant — declaration order *is* the scale. No other change is needed.

## What lands in the output

The `rel_judge` record carries `labels`, one entry per presented subtopic, alongside `qrel_labels`, the official label for each of those subtopics.

```json
{"stage": "rel_judge", "topic_id": "20", "docid": "clueweb09-en0001-01-00001",
 "labels": [{"subtopic": 1, "label": "NeedSatisfied", "evidence": "verbatim quotation"},
            {"subtopic": 2, "label": "NotAddressed", "evidence": ""}],
 "qrel_labels": {"1": 1, "2": 0}, "reason": "..."}
```

Two things to know when reading these records.

`qrel_labels` has an entry for **every** presented subtopic, using 0 where the qrels record nothing. These collections record nonrelevance per document rather than per subtopic, so listing only the graded rows would hide every assessor "no" and make agreement look perfect by construction.

The record does **not** say which scale produced the labels, so when reading older logs, check the run's configuration rather than assuming. Today the label values happen to tell the scales apart, since none of the values coincide, but that is a property of the current vocabularies and not something to rely on.

The cheapest fix is to name the scale in `name` (the experiment name), which is logged as `session_name` on every row — no extra field, no lookup. `custom_settings` works too if the name is already carrying its own meaning.

The record also carries no document-level label. Deciding that, say, a document counts as relevant when at least one subtopic reaches `NeedSatisfied` is an analysis choice, not something to freeze at collection time.

## Example

`scripts/run_session_experiment_subtopic.py` is a complete runner using these settings.
