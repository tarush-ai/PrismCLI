"""Unified paper search: Semantic Scholar first, arXiv fallback."""

from __future__ import annotations

from prism.models import Paper
from prism.search.arxiv import (
    DEFAULT_CATEGORIES,
    extract_keywords_from_papers,
    extract_keywords_from_topic,
    score_relevance,
    search_papers as _arxiv_search,
)
from prism.search.semantic_scholar import search_papers_ss


def search_papers(
    query: str,
    max_results: int = 10,
    categories: list[str] | None = None,
    ss_api_key: str | None = None,
) -> list[Paper]:
    """Search for papers using Semantic Scholar first, falling back to arXiv.

    Semantic Scholar is tried first — no API key required, though providing one
    raises the rate limit. Falls back to arXiv on any SS failure.

    Args:
        query: Free-text search query.
        max_results: Maximum papers to return.
        categories: arXiv category filter (used for arXiv fallback only).
        ss_api_key: Optional Semantic Scholar API key for higher rate limits.

    Returns:
        List of Paper objects.
    """
    try:
        results = search_papers_ss(query, max_results=max_results, api_key=ss_api_key or None)
        if results:
            return results
    except Exception:
        pass  # fall through to arXiv

    return _arxiv_search(query, max_results=max_results, categories=categories)


__all__ = [
    "search_papers",
    "DEFAULT_CATEGORIES",
    "score_relevance",
    "extract_keywords_from_topic",
    "extract_keywords_from_papers",
]
