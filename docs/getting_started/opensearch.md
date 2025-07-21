# How to set up an OpenSearch client

See [OpenSearch Documentation](https://docs.opensearch.org/docs/latest/about/) to learn how to index the corpus of your test collection.

Once you indexed the corpus using opensearch, then set `ToolDescription` in `ExperimentalSettings` in the runner scripts in `scripts` folder. You can set `port` and `description` too.

```
tools=[
    ToolDescription(
        name="opensearch",
        ranking_model="bm25",
        index_name="aquaint_bm25",
        port=9200,
        description="It allows you to perform searches using keywords only and employs the BM25 ranking model to order results.",
    )
```

## How to compare multiple ranking models

You can list multiple `ToolDescription` in `ExperimentalSettings`. Then the experiment will repeat the whole process across the ranking models.

```
tools=[
    ToolDescription(
        name="opensearch",
        ranking_model="bm25",
        index_name="aquaint_bm25",
        port=9200,
        description="It allows you to perform searches using keywords only and employs the BM25 ranking model to order results.",
    ),
    ToolDescription(
        name="opensearch",
        ranking_model="dpr",
        encode_model="sentence-transformers/msmarco-distilbert-base-tas-b",
        index_name="aquaint_dpr",
        port=9200,
        description="It allows you to perform searches using keywords only and employs the DPR ranking model to order results.",
    ),
    ToolDescription(
        name="opensearch",
        ranking_model="splade",
        encode_model="naver/splade-cocondenser-ensembledistil",
        index_name="aquaint_splade",
        port=9200,
        description="It allows you to perform searches using keywords only and employs the Splade ranking model to order results.",
    ),
```
