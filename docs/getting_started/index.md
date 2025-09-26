# Getting Started

## Prerequisites

- A test collection accessible via [ir_datasets](datasets.md)
- [LLMs](llms.md) (OpenAI, Gemini, or local models via `ollama`)
- [OpenSearch client](opensearch.md) that indexed the corpus of the collection
    - Note that an opensearch client is not needed if you're only interested in `query` stage in the experiment.

## Install `geniie-lab`

```
git clone https://github.com/geniie-lab/geniie-lab.git
python -m venv venv
source venv/bin/activate
(venv) python -m pip install -U pip
(venv) python -m pip install -r requirements.txt
(venv) python -m pip install -e .
```

## Set environmental variables

Define these variables in `.env` file.

For OpenSearch client:
```
OPENSEARCH_ADMIN_USER="admin"
OPENSEARCH_ADMIN_PASS="[your password]"
```

For OpenAI LLMs:
```
OPENAI_API_KEY="[your key]"
```

For Azure OpenAI LLMs:
```
AZURE_API_VERSION="[your api version]"
AZURE_ENDPOINT="[your endpoint]"
AZURE_API_KEY="[your key]"
```

For Gemini LLMs:
```
GEMINI_API_KEY="[your key]"
```


## Edit your experimental settings

See [Experiments](../experiments/index.md) to learn how to configure experimental settings. We recommend to start with reviewing the [common settings](../experiments/common_settings.md).

```
scripts/run_session_experiment.py
scripts/run_repetition_experiment.py
scripts/run_agentic_experiment.py
```

## Run the experiment

```
python scripts/run_session_experiment.py
python scripts/run_repetition_experiment.py
python scripts/run_agentic_experiment.py
```

## Save the outputs and logs

```
python scripts/run_session_experiment.py >logs/my_session_experiment.out 2>logs/my_session_experiment.log
python scripts/run_repetition_experiment.py >logs/my_repetition_experiment.out 2>logs/my_repetition_experiment.log
python scripts/run_agentic_experiment.py >logs/my_agentic_experiment.out 2>logs/my_agentic_experiment.log
```
