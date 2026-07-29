"""Unit tests for the LLM helpers with injected dependencies."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from dependencies import LLMDependencies
from llm import generate_response, generate_response_stream


def test_generate_response_uses_injected_client_and_model():
    """The response helper should call the injected client and model."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value.choices = [
        SimpleNamespace(
            message=SimpleNamespace(content="Injected response"),
        )
    ]

    dependencies = LLMDependencies(client=fake_client, model="test-model")

    result = generate_response(
        "What does this do?",
        "Code context",
        "Previous conversation",
        dependencies=dependencies,
    )

    assert result == "Injected response"
    fake_client.chat.completions.create.assert_called_once()
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "test-model"


def test_generate_response_stream_uses_injected_client_and_model():
    """The streaming helper should also use injected dependencies."""
    fake_chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="streamed"))]
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter([fake_chunk])

    dependencies = LLMDependencies(client=fake_client, model="stream-model")

    result = "".join(
        generate_response_stream(
            "What does this do?",
            "Code context",
            "Previous conversation",
            dependencies=dependencies,
        )
    )

    assert result == "streamed"
    fake_client.chat.completions.create.assert_called_once()
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "stream-model"
