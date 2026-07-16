from unittest.mock import patch, MagicMock
from handlers.explain_function import can_handle, handle
from handlers import route_command


def test_can_handle_explain_function():
    # Valid intents
    assert can_handle("explain this function") is True
    assert can_handle("explain the function") is True
    assert can_handle("explain function load_data") is True
    assert can_handle("EXPLAIN THIS FUNCTION") is True

    # Invalid intents
    assert can_handle("show me functions") is False
    assert can_handle("query codebase for preprocess") is False
    assert can_handle("explain what this is") is False


@patch("handlers.explain_function.retrieve")
@patch("handlers.explain_function._get_client")
@patch("handlers.explain_function._get_model")
def test_handle_explain_function(mock_get_model, mock_get_client, mock_retrieve):
    mock_retrieve.return_value = "def load_data(path):\n    return pd.read_csv(path)"

    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="This function loads a CSV data file."))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client
    mock_get_model.return_value = "mock-model"

    res = handle("explain function load_data", "session_123")
    assert res == "This function loads a CSV data file."
    mock_retrieve.assert_called_once_with("explain function load_data")
    mock_client.chat.completions.create.assert_called_once()


@patch("handlers.can_explain")
@patch("handlers.handle_explain")
def test_route_command(mock_handle, mock_can_handle):
    # Test match
    mock_can_handle.return_value = True
    mock_handle.return_value = "Custom explanation response"

    res = route_command("explain this function", "session_123")
    assert res == "Custom explanation response"
    mock_handle.assert_called_once_with("explain this function", "session_123")

    # Test no match
    mock_can_handle.return_value = False
    res2 = route_command("hello world", "session_123")
    assert res2 is None
