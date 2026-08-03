"""Centralized vector store wrapper for Qdrant operations."""

from qdrant_client import QdrantClient, models as qdrant_models
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_SIMILARITY_THRESHOLD,
)
from logger import logger

# Centralized client instance
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def query_points(
    vector: list[float],
    limit: int,
    query_filter: qdrant_models.Filter | None = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
    score_threshold: float = QDRANT_SIMILARITY_THRESHOLD,
):
    """Execute vector similarity search on Qdrant collection."""
    return client.query_points(
        collection_name=collection_name,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold,
    ).points


def get_collection_info(collection_name: str = QDRANT_COLLECTION_NAME):
    """Retrieve metadata and statistics for a Qdrant collection."""
    return client.get_collection(collection_name)


def close_connection():
    """Safely close active Qdrant client connections."""
    try:
        client.close()
        logger.info("Qdrant client connection closed successfully.")
    except Exception as e:
        logger.error("Error during Qdrant client connection cleanup", exc_info=True)
        