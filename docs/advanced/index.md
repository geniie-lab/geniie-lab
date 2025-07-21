# Advanced Topics

Although the [parameters](../parameters.md) in `ExperimentalSettings` allow researchers to configure many aspects of the experiment, one might wish to control the experiment in depth. Here, we illustrate some directions.

## Responses
If you would like to change the format of GII reponses, edit `geniie-lab/response.py`, where structured outputs are defined for LLMs. You can add/remove fields using pydantic format.

```
class Clicks(BaseModel):
    """
    A model for selecting multiple documents from search results.
    """
    ranking_list: List[int] = Field(
        ...,
        title="ranking_list",
        description=(
            "The ranking number of the documents in the result to examine the full text. "
        )
    )
    reason: str = Field(
        ...,
        title="reason",
        description="A brief explanation for selecting these documents."
    )
```

Also, make sure your change is reflected by the `output.py` in `geniie-lab/dataclasses` for final outputs of the experiment.

## Dataclasses
If you want to change any part of I/O in the program, edit dataclass files in `geniie-lab/dataclasses`. We have several files of dataclasses in different categories.

When you changed the fields in the dataclass files, make sure that you also edit the output variables in experiment files in `geniie-lab/experiments`. You might need to change several parts like the following in the experiment files.

```
output = QueryExperimentOutput(
    session_name = settings.name,
    model = model.name,
    task = settings.task.name,
    dataset = settings.topicset.name,
    topic_id = state.topic.id,
    query = state.query.query,
    start = settings.task.start_offset,
    size = settings.task.serp_size
)
```

## Add a new LLM
If you would like to add a new LLM, look into files in `geniie-lab/services/llm` folders and add `your_llm_service.py` based on other llm service files. Then, update `llm_service_factory.py` to add an LLM choice which can then be specified in `ModelDescription` (see [Common Settings](../experiments/common_settings.md)).


## Add a new ranking model in OpenSearch
If you would like to add a new ranking model in OpenSearch, look into files in `geniie-lab/services/opensearch` folders and add `your_opensearch_service.py` based on other opensearcch service files. Then, update `opensearch_service_factory.py` to add a ranking model choice which can then be specified in `ToolDescription` (see [Common Settings](../experiments/common_settings.md)).

## Use a dataset that is not available in `ir_datasets`
If you would like to use a dataset that is not available in `ir_datasets`, you can write a data bridge class in `geniie-lab/databridges` folder. See `geniie-lab/databridges/amazon_esci/README.md` as an example. Then, you need to load `import geniie_lab.databridges.amazon_esci.amazon_esci` in the runner scripts. See `geniie-lab/scripts/run_session_experiment_esci.py` as an example.

## Change the search stage sequence in Agentic Experiment
If you would like to change the sequence of search stages in Agentic Experiemnt, look into `ExperimentRunner` class in `geniie-lab/experiments/agentic_experiemnt.py`.

```
self.action_stage_map = {
    Action.SUBMIT_NEW_QUERY: ["query", "ranking", "click", "relevance", "next_action"],
    Action.CLICK_DOCUMENT: ["click", "relevance", "next_action"],
    Action.GO_NEXT_RESULT_PAGE: ["ranking", "click", "relevance", "next_action"],
}
```

## Add a new experiment
If you would like to change the logic of any of search stages, or the whole experiment, it might make sense to create a new experiment. This is a lot of work but the basic steps are as follows.

1. Create a new file such as `my_experiment.py` in `geniie-lab/experiments` folder based on other experiment files.
    1. Edit an existing stage class or add a new class in `my_experiment.py`.
    1. Edit `ExperimentRunner` class in `my_experiment.py` to control the logics of the whole experiment.
1. Create a new file such as `run_my_experiment.py` in `geniie-lab/scripts` folder based on other runner scripts.
    1. Replace `from geniie_lab.experiments.session_experiment import ExperimentRunner` to `from geniie_lab.experiments.my_experiment import ExperimentRunner`
    1. Edit `ExperimentalSettings` as needed
