import pytest
from unittest.mock import MagicMock, patch

from dependencies import BackendDependencies, IndexingDependencies, LLMDependencies, RetrievalDependencies
from main import shutdown_event


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_shutdown_event():
    """Verify that the shutdown event handler calls the close method

    on the Qdrant client connection.
    """
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock()

    mock_qdrant = MagicMock()

    fake_backend = BackendDependencies(
        retrieval=RetrievalDependencies(
            client=mock_qdrant,
            embedder=mock_embedder,
        ),
        llm=LLMDependencies(
            client=MagicMock(),
            model="test-model",
        ),
        indexing=IndexingDependencies(
            client=mock_qdrant,
            embedder=mock_embedder,
        ),
    )

    with patch("dependencies.get_backend_dependencies", return_value=fake_backend):
        await shutdown_event()
        mock_qdrant.close.assert_called_once()
