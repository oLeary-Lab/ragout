from service.config import Settings
from service.providers.base import EmbeddingsProvider, LLMProvider, Reranker
from service.providers.embeddings import OpenAIEmbeddings
from service.providers.llm import OpenAILLM
from service.providers.reranker import CohereReranker


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        return OpenAILLM(api_key=settings.openai_api_key, model=settings.llm_model)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def build_embeddings(settings: Settings) -> EmbeddingsProvider:
    if settings.embeddings_provider == "openai":
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key, model=settings.embeddings_model
        )
    raise ValueError(f"Unknown embeddings provider: {settings.embeddings_provider}")


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker_provider == "cohere":
        return CohereReranker(
            api_key=settings.cohere_api_key, model=settings.rerank_model
        )
    raise ValueError(f"Unknown reranker provider: {settings.reranker_provider}")
