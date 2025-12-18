# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Embedding providers for RAG systems.

Supports:
- OpenAI embeddings (via langchain_openai)
- Dashscope embeddings (Alibaba Cloud)
- Google Gemini embeddings (gemini-embedding-001) via google-genai SDK
"""

import logging
import os
from typing import Any, List, Sequence

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

logger = logging.getLogger(__name__)


class DashscopeEmbeddings:
    """OpenAI-compatible embeddings wrapper for Dashscope."""

    def __init__(self, **kwargs: Any) -> None:
        self._client: OpenAI = OpenAI(
            api_key=kwargs.get("api_key", ""), base_url=kwargs.get("base_url", "")
        )
        self._model: str = kwargs.get("model", "")
        self._encoding_format: str = kwargs.get("encoding_format", "float")

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Internal helper performing the embedding API call."""
        clean_texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not clean_texts:
            return []
        resp = self._client.embeddings.create(
            model=self._model,
            input=clean_texts,
            encoding_format=self._encoding_format,
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        """Return embedding for a given text."""
        embeddings = self._embed([text])
        return embeddings[0] if embeddings else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for multiple documents (LangChain interface)."""
        return self._embed(texts)


class GeminiEmbeddings:
    """
    Google Gemini embeddings using the google-genai SDK.

    Model: gemini-embedding-001
    Default dimensions: 1536 (can be 768, 1536, or 3072)

    Environment variables:
        GEMINI_API_KEY or api_key parameter: Gemini API key
    """

    DEFAULT_MODEL = "gemini-embedding-001"
    DEFAULT_DIMENSIONS = 1536  # Balanced choice for quality and storage

    def __init__(self, **kwargs: Any) -> None:
        self._api_key: str = kwargs.get("api_key", "") or os.getenv("GEMINI_API_KEY", "")
        self._model: str = kwargs.get("model", self.DEFAULT_MODEL)
        self._dimensions: int = kwargs.get("dimensions", self.DEFAULT_DIMENSIONS)

        if not self._api_key:
            logger.warning(
                "Gemini API key is not set. Set GEMINI_API_KEY environment variable."
            )

        # Initialize Google GenAI client (new unified SDK)
        try:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=self._api_key)
            self._types = types
        except ImportError:
            raise ImportError(
                "google-genai package is required for Gemini embeddings. "
                "Install with: pip install google-genai"
            )

    def _embed(self, texts: Sequence[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            task_type: Task type for embedding optimization
                - RETRIEVAL_QUERY: For search queries
                - RETRIEVAL_DOCUMENT: For documents to be searched
                - SEMANTIC_SIMILARITY: For comparing text similarity
                - CLASSIFICATION: For text classification
                - CLUSTERING: For text clustering

        Returns:
            List of embedding vectors
        """
        clean_texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not clean_texts:
            return []

        embeddings = []
        for text in clean_texts:
            try:
                # Use the new google-genai SDK API
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=text,
                    config=self._types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._dimensions,
                    ),
                )
                # Extract embedding from result
                if hasattr(result, 'embeddings') and result.embeddings:
                    embedding = result.embeddings[0].values
                else:
                    embedding = result.embedding
                embeddings.append(list(embedding))
            except Exception as e:
                logger.error(f"Failed to generate Gemini embedding: {e}")
                # Return zero vector on error
                embeddings.append([0.0] * self._dimensions)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Return embedding for a query text (optimized for search)."""
        embeddings = self._embed([text], task_type="RETRIEVAL_QUERY")
        return embeddings[0] if embeddings else [0.0] * self._dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for documents (optimized for being searched)."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")


# Embedding dimension defaults for common models
EMBEDDING_DIMENSIONS = {
    # OpenAI models
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-v4": 2048,
    # Google Gemini models (default to 1536 for balanced quality/storage)
    "gemini-embedding-001": 1536,
    "embedding-001": 768,  # Legacy
    # Dashscope models
    "text-embedding-v1": 1536,
    "text-embedding-v2": 1536,
}


def get_embedding_dimension(model_name: str, explicit_dim: int = 0) -> int:
    """
    Get embedding dimension for a model.

    Args:
        model_name: Name of the embedding model
        explicit_dim: Explicitly configured dimension (takes precedence)

    Returns:
        Embedding dimension
    """
    if explicit_dim > 0:
        return explicit_dim
    return EMBEDDING_DIMENSIONS.get(model_name, 1536)  # Default to 1536


def create_embedding_model(
    provider: str,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    dimensions: int = 0,
) -> Any:
    """
    Create an embedding model instance based on provider.

    Args:
        provider: Embedding provider (openai, dashscope, gemini)
        model: Model name
        api_key: API key
        base_url: Base URL for API (for OpenAI-compatible providers)
        dimensions: Embedding dimensions

    Returns:
        Embedding model instance
    """
    provider_lower = provider.lower()

    kwargs = {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "encoding_format": "float",
        "dimensions": dimensions or get_embedding_dimension(model),
    }

    if provider_lower == "openai":
        return OpenAIEmbeddings(**kwargs)
    elif provider_lower == "dashscope":
        return DashscopeEmbeddings(**kwargs)
    elif provider_lower == "gemini" or provider_lower == "google":
        return GeminiEmbeddings(
            api_key=api_key,
            model=model or GeminiEmbeddings.DEFAULT_MODEL,
            dimensions=dimensions or GeminiEmbeddings.DEFAULT_DIMENSIONS,
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            "Supported providers: openai, dashscope, gemini"
        )
