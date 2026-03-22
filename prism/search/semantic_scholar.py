"""Semantic Scholar paper search via direct HTTP API."""

from __future__ import annotations

import httpx

from prism.models import Paper

_SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_FIELDS = (
    "title,abstract,authors,year,externalIds,"
    "citationCount,openAccessPdf,fieldsOfStudy"
)


def _item_to_paper(item: dict) -> Paper | None:
    title = (item.get("title") or "").strip()
    if not title:
        return None

    authors = [a.get("name", "") for a in (item.get("authors") or [])]
    year = item.get("year")
    published = f"{year}-01-01" if year else ""

    ext = item.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv", "")
    paper_id = item.get("paperId", "")

    oa = item.get("openAccessPdf") or {}
    url = oa.get("url", "")
    if not url:
        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        elif paper_id:
            url = f"https://www.semanticscholar.org/paper/{paper_id}"

    return Paper(
        id=paper_id or arxiv_id,
        title=title,
        authors=authors,
        abstract=(item.get("abstract") or "").strip(),
        url=url,
        published=published,
        arxiv_id=arxiv_id,
        citation_count=item.get("citationCount") or 0,
    )


def search_papers_ss(
    query: str,
    max_results: int = 10,
    api_key: str | None = None,
) -> list[Paper]:
    """Search Semantic Scholar for papers matching a query.

    Args:
        query: Free-text search query.
        max_results: Maximum papers to return (capped at 100 per API limits).
        api_key: Optional Semantic Scholar API key for higher rate limits.

    Returns:
        List of Paper objects. Empty list on failure.

    Raises:
        RuntimeError: On non-retryable HTTP errors.
    """
    headers: dict[str, str] = {"User-Agent": "PrismCLI/1.0 (research tool)"}
    if api_key:
        headers["x-api-key"] = api_key

    params: dict[str, object] = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": _SS_FIELDS,
        "fieldsOfStudy": "Computer Science",
    }

    resp = httpx.get(
        _SS_SEARCH_URL,
        params=params,
        headers=headers,
        timeout=30,
    )
    if resp.status_code == 200:
        papers = []
        for item in resp.json().get("data", []):
            p = _item_to_paper(item)
            if p:
                papers.append(p)
        return papers

    raise RuntimeError(
        f"Semantic Scholar HTTP {resp.status_code}: {resp.text[:200]}"
    )
