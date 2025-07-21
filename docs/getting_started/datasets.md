# How to set a dataset

Currently, `geniie-lab` expects to access datasets via `ir_datasets`.

Once you can access the test collection via `ir_datastes`, configure `TopicDescription` and `CorpusDescription` in `ExperimentalSettings` in the runner scripts in `scripts` folder.

```
my_settings = ExperimentSettings(
    ...
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
```