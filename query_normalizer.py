"""query_normalizer.py — Query Normalization Layer for DevWhisper.

Standardizes user queries before semantic search by normalizing whitespace, punctuation, and capitalization.
"""

import re
import string

class QueryNormalizer:
    """Normalizes raw natural language queries before retrieval pipeline processing."""

    def normalize_whitespace(self, query: str) -> str:
        """Collapse redundant tabs, newlines, and multi-spaces into single spaces."""
        if not query:
            return ""
        return re.sub(r"\s+", " ", query).strip()

    def normalize_punctuation(self, query: str) -> str:
        """Remove leading/trailing and redundant punctuation surrounding query terms."""
        if not query:
            return ""
        # Strip enclosing quotes and surrounding punctuation while preserving underscores/dots in identifiers
        query = re.sub(r'^[^\w\s()_.\-]+|[^\w\s()_.\-]+$', "", query).strip()
        # Clean repetitive punctuation marks
        query = re.sub(r'([!?,;:\-\.])\1+', r'\1', query)
        return query

    def normalize_capitalization(self, query: str) -> str:
        """Standardize capitalization for non-code words while preserving potential code identifiers."""
        if not query:
            return ""
        tokens = query.split(" ")
        normalized_tokens = []
        for token in tokens:
            # Preserve camelCase, PascalCase, or snake_case identifiers
            if "_" in token or (not token.islower() and not token.isupper()):
                normalized_tokens.append(token)
            else:
                normalized_tokens.append(token.lower())
        return " ".join(normalized_tokens)

    def normalize(self, query: str) -> str:
        """Apply full normalization pipeline: whitespace, punctuation, and capitalization."""
        if not query:
            return ""
        q = self.normalize_whitespace(query)
        q = self.normalize_punctuation(q)
        q = self.normalize_capitalization(q)
        return q

# Global normalizer instance
normalizer = QueryNormalizer()

def normalize_query(query: str) -> str:
    """Convenience function for query normalization layer."""
    return normalizer.normalize(query)
