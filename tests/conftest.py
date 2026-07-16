import sys
from unittest.mock import MagicMock

# Mock SentenceTransformer to prevent downloading/loading the model during tests
mock_sentence_transformers = MagicMock()

class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):
        mock_vector = MagicMock()
        mock_vector.tolist.return_value = [0.0] * 384
        return mock_vector

mock_sentence_transformers.SentenceTransformer = MockSentenceTransformer
sys.modules['sentence_transformers'] = mock_sentence_transformers
