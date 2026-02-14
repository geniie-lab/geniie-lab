from geniie_lab.services.llm.gemini_llm_service import GeminiLLMService
from geniie_lab.services.llm.llm_service_protocol import LLMServiceProtocol
from geniie_lab.services.llm.ollama_llm_service import OllamaLLMService
from geniie_lab.services.llm.openai_llm_service import OpenAILLMService
from geniie_lab.services.llm.azure_llm_service import AzureOpenAILLMService
from geniie_lab.services.llm.openrouter_llm_service import OpenRouterLLMService
from geniie_lab.services.llm.vllm_llm_service import VllmLLMService

class LLMServiceFactory:
    def create_llm_service(self, genai_type: str) -> LLMServiceProtocol:
        if genai_type == "gemini":
            return GeminiLLMService()
        elif genai_type == "ollama":
            return OllamaLLMService()
        elif genai_type == "openai":
            return OpenAILLMService()
        elif genai_type == "azure":
            return AzureOpenAILLMService()
        elif genai_type == "openrouter":
            return OpenRouterLLMService()
        elif genai_type == "vllm":
            return VllmLLMService()
        else:
            raise ValueError(f"Unknown genai_type: {genai_type}")