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

    def extract_filters(self, query: str) -> tuple[str, dict[str, str]]:
        """
        Extract structured filter directives such as 'file:app.py' or 'type:function'
        from query strings, returning the clean query text and extracted filter dictionary.
        """
        if not query:
            return "", {}

        extracted_filters = {}
        # Match pattern key:value
        pattern = r'\b(file|type|symbol|repo):([a-zA-Z0-9_.\-]+)\b'
        
        matches = re.findall(pattern, query, re.IGNORECASE)
        for key, val in matches:
            canonical_key = key.lower()
            if canonical_key == "file":
                extracted_filters["file"] = val
            elif canonical_key == "type":
                extracted_filters["symbol_type"] = val
            elif canonical_key == "symbol":
                extracted_filters["symbol_name"] = val

        # Strip directives out of search query
        clean_query = re.sub(pattern, "", query).strip()
        clean_query = self.normalize_whitespace(clean_query)
        return clean_query, extracted_filters

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

def extract_query_filters(query: str) -> tuple[str, dict[str, str]]:
    """Convenience helper to extract search filters and normalize the query."""
    return normalizer.extract_filters(query)
