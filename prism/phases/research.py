"""Background research phase: deeper synthesis, no researcher interruption."""

from __future__ import annotations

from rich.markdown import Markdown

from prism.models import Investigation
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
    extract_keywords_from_topic,
    search_papers,
)


def run_research_phase(
    investigation: Investigation,
    ai: AIClient,
) -> Investigation:
    """Run the background research phase.

    Performs a deeper keyword-derived search, auto-ingests relevant new papers
    (no researcher selection prompt — they already chose papers in observe), then
    synthesises a comprehensive background context.

    Year filter: defaults to last 5 years. Configurable via
    config.observation.year_from.

    Args:
        investigation: The active investigation.
        ai: The AI client.

    Returns:
        Updated investigation with enriched user_context and phase = 'question'.
    """
    from prism.config import get_observation_config, get_ss_api_key

    obs_cfg = get_observation_config()
    ss_api_key = get_ss_api_key()
    year_from: int | None = obs_cfg.get("year_from")
    if year_from is None:
        import datetime
        year_from = datetime.datetime.now().year - 5
    # Category filter — same scope as observation phase
    categories: list[str] = obs_cfg.get("fields_filter") or DEFAULT_CATEGORIES

    print_phase_banner("research")
    print_header(f"Background Research: {investigation.title}")
    console.print(f"[dim]Category scope: {', '.join(categories)}[/]\n")

    # ── Keywords seeded from topic string, papers as supplement ───────────────
    keywords = extract_keywords_from_topic(
        investigation.topic,
        investigation.papers,
        top_n=6,
    )

    # Build queries: exact topic first, then topic + keyword expansion
    queries: list[str] = [investigation.topic]
    if keywords:
        expansion = " ".join(k for k in keywords if k.lower() not in investigation.topic.lower())
        if expansion.strip():
            queries.append(f"{investigation.topic} {expansion[:80]}")
        print_info(f"Search keywords (topic-seeded): {', '.join(keywords)}")

    existing_ids = {p.arxiv_id for p in investigation.papers}
    ingested: list[str] = []

    for query in queries:
        with spinner(f"Searching [{', '.join(categories[:2])}...]: '{query[:50]}'"):
            try:
                results = search_papers(query, max_results=12, categories=categories, ss_api_key=ss_api_key)
                added = 0
                for p in results:
                    if p.arxiv_id in existing_ids:
                        continue
                    # Year filter
                    if p.year() > 0 and p.year() < year_from:
                        continue
                    investigation.papers.append(p)
                    existing_ids.add(p.arxiv_id)
                    added += 1
                if added:
                    ingested.append(f"+{added} from '{query[:40]}'")
            except Exception as e:
                print_warning(f"Search failed for '{query}': {e}")

    if ingested:
        print_info(f"Ingested: {', '.join(ingested)}")
    else:
        print_info("No new papers found beyond initial search.")

    console.print(f"[dim]Total papers: {len(investigation.papers)} (year ≥ {year_from})[/]\n")

    # ── AI synthesis ───────────────────────────────────────────────────────────
    if not investigation.papers:
        print_warning("No papers available. Proceeding with empty context.")
        investigation.phase = "question"
        investigation.touch()
        return investigation

    print_info(f"Synthesising background from {len(investigation.papers)} paper(s)...")

    with spinner("Synthesising literature..."):
        try:
            synthesis = ai.synthesize_papers(investigation.papers, investigation.topic)
        except Exception as e:
            synthesis = (
                f"Background synthesis failed: {e}\n\n"
                f"Papers: {', '.join(p.title for p in investigation.papers[:5])}"
            )

    console.print()
    console.rule("[dim cyan]Background Synthesis[/]", style="dim cyan")
    console.print(Markdown(synthesis))
    console.rule(style="dim cyan")
    console.print()

    investigation.user_context = synthesis
    investigation.phase = "question"
    investigation.touch()

    print_success("Background research complete. Ready to define research question.")
    return investigation
