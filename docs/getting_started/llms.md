# How to set up LLMs

## OpenAI compatible models

Set `OPENAI_API_KEY` in `.env` file

```
OPENAI_API_KEY="[your key]"
```

Set your model name in `ExperimentalSettings` in the runner files in `scripts`. You must also set `token_length` (your model's max input token length); the `system_prompt`, `temperature`, and `top_p` are optional.

```
models=[
    ModelDescription(
        type="openai",
        name="gpt-4.1-mini-2025-04-14",
        token_length=1000000,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    )
```

## Azure OpenAI compatible models

Set `AZURE_API_VERSION`, `AZURE_ENDPOINT`, and `AZURE_API_KEY` in `.env` file

```
AZURE_API_VERSION="[your api version]"
AZURE_ENDPOINT="[your endpoint]"
AZURE_API_KEY="[your key]"
```

Set your model name in `ExperimentalSettings` in the runner files in `scripts`. You must also set `token_length` (your model's max input token length); the `system_prompt`, `temperature`, and `top_p` are optional.

```
models=[
    ModelDescription(
        type="azure",
        name="gpt-4.1-mini",
        token_length=1000000,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    )
```

## Gemini compatible models

Set `GEMINI_API_KEY` in `.env` file

```
GEMINI_API_KEY="[your key]"
```

Set your model name in `ExperimentalSettings` in the runner files in `scripts`

```
models=[
    ModelDescription(
        type="gemini",
        name="gemini-2.0-flash-lite-001",
        token_length=1048576,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    )
```

## OpenRouter compatible models

Set `OPENROUTER_API_KEY` in `.env` file

```
OPENROUTER_API_KEY="[your key]"
```

Set your model name in `ExperimentalSettings` in the runner files in `scripts`

```
models=[
    ModelDescription(
        type="openrouter",
        name="openai/gpt-4.1-mini",
        token_length=1000000,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    )
```

## Groq models

Set `GROQ_API_KEY` in `.env` file

```
GROQ_API_KEY="[your key]"
```

Then, set your model name in `ExperimentalSettings` in the runner files in `scripts`

```
models=[
    ModelDescription(
        type="groq",
        name="openai/gpt-oss-120b",
        token_length=128000,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    )
```

Note: the free tier is limited to 8,000 tokens per minute, which is enough for small pilot runs (e.g. query and ranking stages) but not for full sessions; a paid plan is needed for complete experiments.

## Amazon Bedrock models

Set `BEDROCK_API_KEY` and `BEDROCK_REGION` in `.env` file

```
BEDROCK_API_KEY="[your Amazon Bedrock API key]"
BEDROCK_REGION="us-east-1"
```

You also need to enable model access for the models you want to use in the Amazon Bedrock console of that region.

Then, set your model name in `ExperimentalSettings` in the runner files in `scripts`

```
models=[
    ModelDescription(
        type="bedrock",
        name="openai.gpt-oss-120b",
        token_length=128000,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    )
```

Note: Bedrock model names use dots (`openai.gpt-oss-120b`), not slashes. Structured outputs are handled by `geniie-lab` via prompt-embedded schemas on this provider, and the schema tokens are excluded from the reported token counts.

## Local models via `ollama`

Local models should be loaded via `ollama` at http://localhost:11434/v1.

Then, set your model name in `ExperimentalSettings` in the runner files in `scripts`

```
models=[
    ModelDescription(
        type="ollama",
        name="llama3.3:70b-instruct-q4_K_M",
        token_length=128000,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    )
```

## Local models via `vllm`

Local models should be served by `vllm` at http://localhost:8000/v1, for example:

```
vllm serve openai/gpt-oss-120b --host 0.0.0.0
```

Then, set your model name in `ExperimentalSettings` in the runner files in `scripts`

```
models=[
    ModelDescription(
        type="vllm",
        name="openai/gpt-oss-120b",
        token_length=128000,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    )
```

Note: the model name must match the name the `vllm` server was started with.

## How to compare multiple LLMs

You can list `ModelDescription` in `ExperimentalSettings` as follows. Then the experiment will repeat the whole process across the models.

```
models=[
    ModelDescription(
        type="openai",
        name="gpt-4.1-mini-2025-04-14",
        token_length=1000000,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    ),
    ModelDescription(
        type="gemini",
        name="gemini-2.0-flash-lite-001",
        token_length=1048576,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    ),
    ModelDescription(
        type="ollama",
        name="llama3.3:70b-instruct-q4_K_M",
        token_length=128000,  # set to your model's max input token length
        system_prompt="You're a helpful assistant",
        temperature=0.0,
        top_p=1.0,
    )
]
```
