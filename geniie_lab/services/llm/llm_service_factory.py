# Standard library
import os
from typing import Callable, Dict

# Third-party libraries
from dotenv import load_dotenv
from openai import AzureOpenAI

# Local application imports
from geniie_lab.services.llm.gemini_llm_service import GeminiLLMService
from geniie_lab.services.llm.llm_service_protocol import LLMServiceProtocol
from geniie_lab.services.llm.openai_compatible import OpenAICompatibleLLMService


def _make_azure() -> OpenAICompatibleLLMService:
    return OpenAICompatibleLLMService(client=AzureOpenAI(
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
    ))


# Adding a provider that speaks the OpenAI chat API is one entry here.
_REGISTRY: Dict[str, Callable[[], LLMServiceProtocol]] = {
    "openai": lambda: OpenAICompatibleLLMService(),
    "azure": _make_azure,
    "openrouter": lambda: OpenAICompatibleLLMService(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ),
    "groq": lambda: OpenAICompatibleLLMService(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    ),
    "ollama": lambda: OpenAICompatibleLLMService(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
        tiktoken_by_model=False,
    ),
    "vllm": lambda: OpenAICompatibleLLMService(
        base_url="http://localhost:8000/v1",
        api_key="vllm",  # required, but unused
        tiktoken_by_model=False,
    ),
    "gemini": GeminiLLMService,
}


class LLMServiceFactory:
    def create_llm_service(self, genai_type: str) -> LLMServiceProtocol:
        # The per-provider services used to load .env themselves; keep that
        # behavior for library users who do not call load_dotenv().
        load_dotenv()
        try:
            return _REGISTRY[genai_type]()
        except KeyError:
            raise ValueError(f"Unknown genai_type: {genai_type}")
