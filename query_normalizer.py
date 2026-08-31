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

    #: Directive keyword -> the payload field it filters on.
    #:
    #: Every key here is both stripped from the query text and mapped to a
    #: filter. The two used to be driven by separate lists — ``repo`` was in
    #: the regex, so it was deleted from the query, but had no branch in the
    #: mapping, so it was silently discarded (issue #308). Deriving the
    #: pattern from this table makes that class of mismatch unrepresentable:
    #: a key cannot be stripped unless it is also mapped.
    DIRECTIVE_FIELDS = {
        "file": "file",
        "type": "symbol_type",
        "symbol": "symbol_name",
        "repo": "repository",
    }

    #: Built from DIRECTIVE_FIELDS rather than restated, for the reason above.
    DIRECTIVE_PATTERN = (
        r'\b(' + "|".join(DIRECTIVE_FIELDS) + r'):([a-zA-Z0-9_.\-]+)\b'
    )

    def extract_filters(self, query: str) -> tuple[str, dict[str, str]]:
        """
        Extract structured filter directives from a query string.

        Recognises ``file:app.py``, ``type:function``, ``symbol:retrieve`` and
        ``repo:backend``, returning the query with the directives removed and
        a dict of payload field -> value.

        A directive is only removed from the query if it is also returned as a
        filter. ``repo:`` used to be removed without being returned, so
        ``"show symbol:retrieve repo:devwhisper"`` reduced to the query
        ``"show"`` and searched whichever repository happened to be active,
        with no indication that anything had been ignored.

        Args:
            query: Raw query string, possibly containing directives.

        Returns:
            ``(clean_query, filters)``. Keys in *filters* are payload field
            names, not directive keywords — ``type:`` maps to ``symbol_type``,
            ``repo:`` to ``repository``.
        """
        if not query:
            return "", {}

        extracted_filters = {}
        matches = re.findall(self.DIRECTIVE_PATTERN, query, re.IGNORECASE)
        for key, val in matches:
            field = self.DIRECTIVE_FIELDS[key.lower()]
            extracted_filters[field] = val

        # Strip directives out of search query
        clean_query = re.sub(self.DIRECTIVE_PATTERN, "", query, flags=re.IGNORECASE).strip()
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
