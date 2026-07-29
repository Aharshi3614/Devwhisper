"""Shared dependency factories for DevWhisper backend services."""

from dataclasses import dataclass
from functools import lru_cache
import os

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_OPENAI_COMPATIBLE_MODEL,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY_ENV,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    QDRANT_API_KEY_ENV,
    QDRANT_URL_ENV,
)


@dataclass(frozen=True)
class RetrievalDependencies:
    """Dependencies required to run code retrieval."""

    client: QdrantClient
    embedder: SentenceTransformer


@dataclass(frozen=True)
class LLMDependencies:
    """Dependencies required to call the language model."""

    client: OpenAI
    model: str


@dataclass(frozen=True)
class IndexingDependencies:
    """Dependencies required to build and validate the vector index."""

    client: QdrantClient
    embedder: SentenceTransformer


@dataclass(frozen=True)
class BackendDependencies:
    """Combined backend dependencies used by the FastAPI app."""

    retrieval: RetrievalDependencies
    llm: LLMDependencies
    indexing: IndexingDependencies


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create the shared Qdrant client once per process."""
    return QdrantClient(
        url=os.getenv(QDRANT_URL_ENV),
        api_key=os.getenv(QDRANT_API_KEY_ENV),
    )


@lru_cache(maxsize=1)
def get_retrieval_embedder() -> SentenceTransformer:
    """Create the retrieval embedder used at query time."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


@lru_cache(maxsize=1)
def get_indexing_embedder() -> SentenceTransformer:
    """Create the indexing embedder used during offline indexing."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    """Create the shared OpenAI-compatible client."""
    provider_api_key = os.getenv(LLM_API_KEY_ENV)
    if provider_api_key is None:
        return OpenAI(
            api_key=os.getenv(GROQ_API_KEY_ENV),
            base_url=DEFAULT_LLM_BASE_URL,
        )

    return OpenAI(
        api_key=provider_api_key or os.getenv(GROQ_API_KEY_ENV),
        base_url=os.getenv(LLM_BASE_URL_ENV, DEFAULT_LLM_BASE_URL),
    )


def get_llm_model() -> str:
    """Return the configured LLM model name."""
    explicit_model = os.getenv(LLM_MODEL_ENV)
    if explicit_model:
        return explicit_model

    if os.getenv(LLM_API_KEY_ENV) is None:
        return DEFAULT_GROQ_MODEL
    return DEFAULT_OPENAI_COMPATIBLE_MODEL


@lru_cache(maxsize=1)
def get_retrieval_dependencies() -> RetrievalDependencies:
    """Build retrieval dependencies from the shared factories."""
    return RetrievalDependencies(
        client=get_qdrant_client(),
        embedder=get_retrieval_embedder(),
    )


@lru_cache(maxsize=1)
def get_indexing_dependencies() -> IndexingDependencies:
    """Build indexing dependencies from the shared factories."""
    return IndexingDependencies(
        client=get_qdrant_client(),
        embedder=get_indexing_embedder(),
    )


@lru_cache(maxsize=1)
def get_llm_dependencies() -> LLMDependencies:
    """Build LLM dependencies from the shared factories."""
    return LLMDependencies(
        client=get_llm_client(),
        model=get_llm_model(),
    )


@lru_cache(maxsize=1)
def get_backend_dependencies() -> BackendDependencies:
    """Bundle all backend dependencies used by the FastAPI app."""
    return BackendDependencies(
        retrieval=get_retrieval_dependencies(),
        llm=get_llm_dependencies(),
        indexing=get_indexing_dependencies(),
    )


def reset_dependency_caches() -> None:
    """Clear cached providers, mainly useful in tests."""
    get_qdrant_client.cache_clear()
    get_retrieval_embedder.cache_clear()
    get_indexing_embedder.cache_clear()
    get_llm_client.cache_clear()
    get_retrieval_dependencies.cache_clear()
    get_indexing_dependencies.cache_clear()
    get_llm_dependencies.cache_clear()
    get_backend_dependencies.cache_clear()
