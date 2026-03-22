"""arXiv paper discovery with category filtering and embedding-based relevance scoring."""

from __future__ import annotations

import math
import re
import time

import arxiv


from prism.models import Paper


def _sanitize_query(query: str) -> str:
    """Strip ANSI escape sequences and control characters from a search query."""
    # Remove ANSI escape sequences (e.g. \x1b[31m, \x1b[2J, bare \x1b, etc.)
    query = re.sub(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])?", "", query)
    # Remove remaining ASCII control characters (keep printable + tab/newline)
    query = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", query)
    return query.strip()


# Default ML/AI category scope — keeps cs.LG topics from drifting into hep-th
DEFAULT_CATEGORIES = ["cs.CL", "cs.LG", "cs.AI", "stat.ML"]


def _result_to_paper(result: arxiv.Result) -> Paper:
    arxiv_id = result.entry_id.split("/")[-1]
    if "v" in arxiv_id:
        arxiv_id = arxiv_id.rsplit("v", 1)[0]
    return Paper(
        id=result.entry_id,
        title=result.title.strip(),
        authors=[str(a) for a in result.authors],
        abstract=result.summary.strip(),
        url=result.pdf_url or result.entry_id,
        published=result.published.strftime("%Y-%m-%d") if result.published else "",
        arxiv_id=arxiv_id,
    )


def search_papers(
    query: str,
    max_results: int = 10,
    categories: list[str] | None = None,
) -> list[Paper]:
    """Search arXiv, scoped to the given category list.

    The category filter is ANDed onto the query so topic-level terms drive
    ranking while off-domain fields (physics, chemistry, etc.) are excluded.

    Args:
        query: Free-text search query.
        max_results: Max results before relevance filtering.
        categories: arXiv category strings to restrict to.
                    Defaults to DEFAULT_CATEGORIES (cs.CL/LG/AI + stat.ML).
                    Pass [] to search all categories.

    Returns:
        List of Paper objects, ranked by arXiv relevance.
    """
    query = _sanitize_query(query)
    if not query:
        return []

    cats = categories if categories is not None else DEFAULT_CATEGORIES

    if cats:
        cat_clause = " OR ".join(f"cat:{c}" for c in cats)
        full_query = f"({query}) AND ({cat_clause})"
    else:
        full_query = query

    # Match page_size to max_results so each HTTP request only asks for what we
    # need (default page_size=100 sends max_results=100 to arXiv even for small
    # queries, which triggers 503s under load).
    client = arxiv.Client(
        page_size=max(max_results, 1),
        delay_seconds=3.0,
        num_retries=5,
    )
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return [_result_to_paper(r) for r in client.results(search)]
        except Exception as exc:
            msg = str(exc)
            if any(code in msg for code in ("500", "503", "502", "504")):
                last_exc = exc
                time.sleep(4 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"arXiv search failed after retries: {last_exc}")


def get_paper(arxiv_id: str) -> Paper | None:
    client = arxiv.Client()
    results = list(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    return _result_to_paper(results[0]) if results else None


def search_by_category(categories: list[str], max_results: int = 20) -> list[Paper]:
    if not categories:
        return []
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    client = arxiv.Client()
    search = arxiv.Search(
        query=cat_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return [_result_to_paper(r) for r in client.results(search)]


# ── Embedding-based relevance scoring ─────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na * nb > 0 else 0.0


def score_relevance(
    papers: list[Paper],
    topic: str,
    api_key: str,
    model: str = "text-embedding-3-small",
) -> list[tuple[Paper, float]]:
    """Score each paper by cosine similarity to the investigation topic.

    Uses OpenAI text-embedding-3-small. Batches all paper texts in a single
    API call to minimise latency and cost.

    Args:
        papers: Papers to score.
        topic: The investigation topic string (source of truth for relevance).
        api_key: OpenAI API key.
        model: Embedding model name.

    Returns:
        List of (paper, score) tuples, same order as input.
        Scores are cosine similarities in [0, 1].
    """
    from openai import OpenAI

    if not papers:
        return []

    client = OpenAI(api_key=api_key)

    # Single API call: topic + all papers
    texts = [topic] + [f"{p.title}. {p.abstract[:400]}" for p in papers]
    resp = client.embeddings.create(model=model, input=texts)
    embeddings = [d.embedding for d in resp.data]

    topic_emb = embeddings[0]
    paper_embs = embeddings[1:]

    return [(paper, _cosine(topic_emb, emb)) for paper, emb in zip(papers, paper_embs)]


# ── Keyword extraction (seeded from topic, papers as supplement) ──────────────

def extract_keywords_from_topic(topic: str, papers: list[Paper], top_n: int = 6) -> list[str]:
    """Derive search keywords seeded from the investigation topic string.

    The topic's own words take priority. Paper corpus terms are used only to
    supplement if the topic alone yields fewer than top_n terms.

    Args:
        topic: The original investigation topic string (primary seed).
        papers: Retrieved papers (secondary signal only).
        top_n: Target number of keyword phrases.

    Returns:
        List of keyword strings, topic-first.
    """
    import re
    from collections import Counter

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "this",
        "that", "these", "those", "it", "its", "we", "our", "their",
        "show", "shows", "propose", "proposed", "present", "paper", "work",
        "using", "based", "model", "models", "method", "methods", "approach",
        "results", "data", "study", "also", "than", "more", "which", "such",
        "new", "use", "used", "first", "two", "large", "high", "both", "each",
    }

    # 1. Extract meaningful words from the topic string itself
    topic_words = re.findall(r"\b[a-z][a-z\-]{2,}\b", topic.lower())
    topic_terms = [w for w in topic_words if w not in stop_words]

    # Build bigrams from the topic (highest signal)
    topic_bigrams = [
        f"{topic_terms[i]} {topic_terms[i+1]}"
        for i in range(len(topic_terms) - 1)
    ]

    keywords: list[str] = list(dict.fromkeys(topic_bigrams + topic_terms))[:top_n]

    # 2. Supplement from paper corpus only if we don't have enough
    if len(keywords) < top_n and papers:
        word_counts: Counter = Counter()
        for paper in papers:
            text = (paper.title + " " + paper.abstract).lower()
            for w in re.findall(r"\b[a-z][a-z\-]{3,}\b", text):
                if w not in stop_words and w not in keywords:
                    word_counts[w] += 1
        for w, _ in word_counts.most_common(top_n * 2):
            if w not in keywords:
                keywords.append(w)
            if len(keywords) >= top_n:
                break

    return keywords[:top_n]


# Keep old name as alias so other callers don't break
def extract_keywords_from_papers(papers: list[Paper], top_n: int = 5) -> list[str]:
    """Legacy: extract keywords from papers only. Prefer extract_keywords_from_topic."""
    return extract_keywords_from_topic("", papers, top_n=top_n)
