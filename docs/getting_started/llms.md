# How to set up LLMs

## OpenAI compatible models

Set `OPENAI_API_KEY` in `.env` file

```
OPENAI_API_KEY="[your key]"
```

Set your model name in `ExperimentalSettings` in the runner files in `scripts`. You can set the `system_prompt` and `temperature` too.

```
models=[
    ModelDescription(
        type="openai",
        name="gpt-4.1-mini-2025-04-14",
        system_prompt="You're a helpful assistant",
        temperature=0.0,
    )
```

## Azure OpenAI compatible models

Set `AZURE_API_VERSION`, `AZURE_ENDPOINT`, and `AZURE_API_KEY` in `.env` file

```
AZURE_API_VERSION="[your api version]"
AZURE_ENDPOINT="[your endpoint]"
AZURE_API_KEY="[your key]"
```

Set your model name in `ExperimentalSettings` in the runner files in `scripts`. You can set the `system_prompt` and `temperature` too.

```
models=[
    ModelDescription(
        type="azure",
        name="gpt-4.1-mini",
        system_prompt="You're a helpful assistant",
        temperature=0.0,
    )
```

## Gemini compatible models

Set `GENAI_API_KEY` in `.env` file

```
GENAI_API_KEY="[your key]"
```

Set your model name in `ExperimentalSettings` in the runner files in `scripts`

```
models=[
    ModelDescription(
        type="gemini",
        name="gemini-2.0-flash-lite-001",
        system_prompt="You're a helpful assistant",
        temperature=0.0,
    )
```

## Local models via `ollama`

Local models should be loaded via `ollama` at http://localhost:11434/v1.

Then, set your model name in `ExperimentalSettings` in the runner files in `scripts`

```
models=[
    ModelDescription(
        type="ollama",
        name="llama3.3:70b-instruct-q4_K_M",
        system_prompt="You're a helpful assistant",
        temperature=0.0,
    )
```

## How to compare multiple LLMs

You can list `ModelDescription` in `ExperimentalSettings` as follows. Then the experiment will repeat the whole process across the models.

```
models=[
    ModelDescription(
        type="openai",
        name="gpt-4.1-mini-2025-04-14",
        system_prompt="You're a helpful assistant",
        temperature=0.0,
    ),
    ModelDescription(
        type="gemini",
        name="gemini-2.0-flash-lite-001",
        system_prompt="You're a helpful assistant",
        temperature=0.0,
    ),
    ModelDescription(
        type="ollama",
        name="llama3.3:70b-instruct-q4_K_M",
        system_prompt="You're a helpful assistant",
        temperature=0.0,
    )
]
```