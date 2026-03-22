"""Observation phase: category-scoped arXiv discovery with embedding relevance filtering."""

from __future__ import annotations

import typer
from rich.markdown import Markdown

from prism.models import Investigation, Paper
from prism.display import (
    console,
    print_header,
    print_info,
    print_phase_banner,
    print_success,
    print_warning,
    spinner,
)
from prism.orchestrator import AIClient
from prism.search import (
    DEFAULT_CATEGORIES,
    score_relevance,
    search_papers,
)

RELEVANCE_REJECT  = 0.35   # papers below this are dropped before display
RELEVANCE_WARN    = 0.40   # batches with median below this get a warning


def _score_label(score: float) -> str:
    """Colour-coded relevance badge."""
    if score >= 0.55:
        return f"[bold green]{score:.2f}[/]"
    if score >= RELEVANCE_REJECT:
        return f"[yellow]{score:.2f}[/]"
    return f"[red]{score:.2f}[/]"


def _print_paper_with_score(paper: Paper, index: int, score: float) -> None:
    badge = _score_label(score)
    console.print(f"  [bold cyan][{index}][/]  [bold white]{paper.title}[/]  [dim]sim={badge}[/]")
    console.print(f"       [dim cyan]{paper.short_authors(3)}[/]  [dim]{paper.published}[/]")
    abstract = paper.abstract[:200].replace("\n", " ")
    if len(paper.abstract) > 200:
        abstract += "..."
    console.print(f"       [dim white]{abstract}[/]")
    console.print(f"       [dim blue underline]{paper.url}[/]")
    console.print()


def run_observe_phase(
    investigation: Investigation,
    ai: AIClient,
    limit: int | None = None,
) -> Investigation:
    """Run the observation phase.

    - Searches arXiv scoped to ML/AI categories (cs.CL, cs.LG, cs.AI, stat.ML)
      to prevent off-domain results.
    - Scores every candidate against the investigation topic using
      text-embedding-3-small cosine similarity.
    - Rejects papers with similarity < 0.35 before presenting them.
    - Warns if the batch median falls below 0.40 (indicates query drift).
    - Default selection is all passing papers — press Enter to accept.

    Args:
        investigation: The active investigation.
        ai: The AI client (also provides the API key for embeddings).
        limit: Max papers to fetch from arXiv (before filtering).

    Returns:
        Updated investigation with papers added and phase = 'research'.
    """
    from prism.config import get_api_key, get_observation_config, get_ss_api_key

    obs_cfg = get_observation_config()
    ss_api_key = get_ss_api_key()
    if limit is None:
        limit = obs_cfg.get("paper_limit", 10)

    # Category filter — use config override if set, else DEFAULT_CATEGORIES
    categories: list[str] = obs_cfg.get("fields_filter") or DEFAULT_CATEGORIES

    print_phase_banner("observation")
    print_header(f"Observing: {investigation.title}")
    console.print(f"[dim]Category scope: {', '.join(categories)}[/]\n")

    from prism.config import get_user_profile
    profile = get_user_profile()
    research_areas = profile.get("research_areas", [])

    queries = [investigation.topic]
    if research_areas:
        queries.append(f"{investigation.topic} {' '.join(research_areas[:2])}")

    raw_papers: list[Paper] = []
    seen_ids: set[str] = set()

    for query in queries[:2]:
        with spinner(f"Searching [{', '.join(categories[:2])}...]: '{query[:55]}'"):
            try:
                results = search_papers(query, max_results=limit, categories=categories, ss_api_key=ss_api_key)
                for p in results:
                    if p.arxiv_id not in seen_ids:
                        raw_papers.append(p)
                        seen_ids.add(p.arxiv_id)
            except Exception as e:
                print_warning(f"arXiv search failed for '{query}': {e}")

    if not raw_papers:
        print_warning("No papers found. Try broadening your topic.")
        return investigation

    # ── Embedding relevance scoring ────────────────────────────────────────────
    api_key = get_api_key()
    scored: list[tuple[Paper, float]] = []

    if api_key:
        with spinner(f"Scoring {len(raw_papers)} papers against topic embedding..."):
            try:
                scored = score_relevance(raw_papers, investigation.topic, api_key)
            except Exception as e:
                print_warning(f"Relevance scoring failed ({e}). Showing all papers unscored.")
                scored = [(p, 1.0) for p in raw_papers]
    else:
        print_warning("No API key — skipping relevance scoring, showing all papers.")
        scored = [(p, 1.0) for p in raw_papers]

    # ── Filter below threshold ─────────────────────────────────────────────────
    passing = [(p, s) for p, s in scored if s >= RELEVANCE_REJECT]
    rejected_count = len(scored) - len(passing)

    if rejected_count:
        print_info(
            f"Filtered out {rejected_count} paper(s) with similarity < {RELEVANCE_REJECT:.2f}."
        )

    if not passing:
        print_warning(
            f"All {len(raw_papers)} papers scored below {RELEVANCE_REJECT:.2f}. "
            "Try a more specific topic or expand the category list."
        )
        return investigation

    # ── Median quality warning ─────────────────────────────────────────────────
    scores_only = [s for _, s in passing]
    median_score = sorted(scores_only)[len(scores_only) // 2]
    if median_score < RELEVANCE_WARN:
        console.print()
        console.print(
            f"[bold yellow]⚠  Low relevance batch[/] — median similarity = "
            f"[bold yellow]{median_score:.2f}[/] (threshold: {RELEVANCE_WARN:.2f})\n"
            f"[dim]The search may have returned loosely related papers. "
            f"Consider refining your topic or adding category filters.[/]"
        )
        console.print()

    # ── Sort by score desc, display ────────────────────────────────────────────
    passing.sort(key=lambda x: x[1], reverse=True)
    print_success(f"Found {len(passing)} relevant papers (median sim={median_score:.2f}).\n")

    for i, (paper, score) in enumerate(passing, 1):
        _print_paper_with_score(paper, i, score)

    # ── Selection — default all ────────────────────────────────────────────────
    console.print(
        "[bold cyan]Select papers to include:[/]\n"
        f"  Press [bold]Enter[/] to include all {len(passing)}\n"
        "  Or enter comma-separated numbers (e.g. [bold]1,3,5[/]) to choose specific papers\n"
        "  Enter [bold]n[/] to skip all\n"
    )

    selected: list[Paper] = []
    while True:
        raw = typer.prompt("Selection", default="a").strip().lower()
        if raw in ("", "a"):
            selected = [p for p, _ in passing]
            break
        elif raw == "n":
            print_info("No papers selected.")
            break
        else:
            try:
                indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
                invalid = [i for i in indices if i < 1 or i > len(passing)]
                if invalid:
                    print_warning(f"Invalid indices: {invalid}. Try again.")
                    continue
                selected = [passing[i - 1][0] for i in indices]
                break
            except ValueError:
                print_warning("Enter numbers separated by commas, Enter for all, or 'n' to skip.")

    if not selected:
        print_info("No papers selected. Continuing.")
        investigation.phase = "research"
        investigation.touch()
        return investigation

    print_success(f"Selected {len(selected)} paper(s).")

    existing_ids = {p.arxiv_id for p in investigation.papers}
    new_papers = [p for p in selected if p.arxiv_id not in existing_ids]
    investigation.papers.extend(new_papers)
    if new_papers:
        print_info(f"Added {len(new_papers)} new paper(s).")

    # ── Theme summary ──────────────────────────────────────────────────────────
    if len(investigation.papers) >= 2:
        console.print()
        with spinner("Analysing key themes..."):
            try:
                summary = ai.analyze_papers(
                    investigation.papers,
                    focus=f"key themes and relevance to: {investigation.topic}",
                )
            except Exception as e:
                summary = f"[Theme analysis unavailable: {e}]"

        console.print()
        console.rule("[dim cyan]Key Themes[/]", style="dim cyan")
        console.print(Markdown(summary))
        console.rule(style="dim cyan")
        console.print()

        if not investigation.user_context:
            investigation.user_context = f"Initial observation summary:\n{summary}"

    investigation.phase = "research"
    investigation.touch()
    print_success("Observation complete. Ready for background research.")
    return investigation
