from geniie_lab.services.llm.gemini_llm_service import GeminiLLMService
from geniie_lab.services.llm.llm_service_protocol import LLMServiceProtocol
from geniie_lab.services.llm.ollama_llm_service import OllamaLLMService
from geniie_lab.services.llm.openai_llm_service import OpenAILLMService
from geniie_lab.services.llm.azure_llm_service import AzureOpenAILLMService

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
        else:
            raise ValueError(f"Unknown genai_type: {genai_type}")