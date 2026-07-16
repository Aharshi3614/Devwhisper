import pytest
from unittest.mock import patch
from main import shutdown_event


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_shutdown_event():
    """Verify that the shutdown event handler calls the close method

    on the Qdrant client connection.
    """
    with patch("main.qdrant_client") as mock_qdrant:
        await shutdown_event()
        mock_qdrant.close.assert_called_once()
