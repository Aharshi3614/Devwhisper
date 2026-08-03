import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

mock_rank_bm25_module = MagicMock()

class MockBM250kapi:
    def __init__(self, corpus):
        self.corpus = corpus
    def get_scores(self, query):
        return [0.0] * len(self.corpus)

mock_rank_bm25_module.BM250kapi = MockBM250kapi
sys.modules['rank_bm25'] = mock_rank_bm25_module
