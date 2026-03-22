"""Conclusion phase: Judge novelty scoring + structured paper generation."""

from __future__ import annotations

import re
from pathlib import Path

import typer
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from prism.taste.base import ModelBackend
from prism.models import Investigation, NoveltyScore
from prism.display import (
    console,
    print_error,
    print_header,
    print_info,
    print_phase_banner,
    print_success,
    print_warning,
    spinner,
)
from prism.orchestrator import AIClient

PRISM_DIR = Path.home() / ".prism"


def _generate_bibtex(investigation: Investigation) -> str:
    entries: list[str] = []
    for paper in investigation.papers:
        first_author = paper.authors[0] if paper.authors else "Unknown"
        last_name = re.sub(r"[^a-z]", "", first_author.split()[-1].lower()) or "author"
        year = paper.published[:4] if paper.published and len(paper.published) >= 4 else "0000"
        title_words = re.findall(r"[A-Za-z]{4,}", paper.title)
        topic_word = title_words[0].lower() if title_words else "paper"
        cite_key = f"{last_name}{year}{topic_word}"
        bib_authors = " and ".join(paper.authors[:6])
        if len(paper.authors) > 6:
            bib_authors += " and others"
        entry = (
            f"@article{{{cite_key},\n"
            f"  title = {{{paper.title}}},\n"
            f"  author = {{{bib_authors}}},\n"
            f"  year = {{{year}}},\n"
            f"  url = {{{paper.url}}},\n"
            f"}}"
        )
        entries.append(entry)
    return "\n\n".join(entries)


def _run_judge_scoring(
    investigation: Investigation,
    backend: ModelBackend,
) -> list[NoveltyScore]:
    """Compare the selected idea against reference papers using the Judge."""
    from prism.config import get_judge_config

    judge_cfg = get_judge_config()
    max_comparisons = judge_cfg.get("comparisons_per_idea", 5)

    # Determine what to judge: selected Thinker idea, or research question as abstract
    selected_idea = investigation.selected_idea()
    if selected_idea:
        idea_title = selected_idea.title
        idea_abstract = selected_idea.abstract
    elif investigation.research_question:
        idea_title = investigation.research_question
        idea_abstract = (
            investigation.conclusion[:400]
            if investigation.conclusion
            else f"Investigation into: {investigation.research_question}"
        )
    else:
        return []

    reference_papers = investigation.papers[:max_comparisons]
    if not reference_papers:
        return []

    scores: list[NoveltyScore] = []
    for paper in reference_papers:
        try:
            score = backend.judge(idea_title, idea_abstract, paper)
            scores.append(score)
        except Exception as e:
            print_warning(f"Judge failed on '{paper.title[:40]}': {e}")

    return scores


def _display_novelty_report(scores: list[NoveltyScore], win_rate: float) -> None:
    """Render Judge scoring results as a rich table."""
    if not scores:
        return

    table = Table(
        title="[bold magenta]Novelty Scoring (Judge)[/]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim magenta",
        show_lines=True,
    )
    table.add_column("Reference Paper", style="white", min_width=30)
    table.add_column("Winner", justify="center", width=10)
    table.add_column("Reasoning", style="dim white", min_width=30)

    for s in scores:
        winner_str = (
            "[bold green]Your Idea[/]" if s.winner == "idea"
            else "[bold yellow]Reference[/]"
        )
        table.add_row(
            s.reference_paper_title[:60],
            winner_str,
            s.judge_reasoning[:120] + ("..." if len(s.judge_reasoning) > 120 else ""),
        )

    console.print(table)
    console.print()

    pct = int(win_rate * 100)
    color = "green" if win_rate >= 0.6 else ("yellow" if win_rate >= 0.4 else "red")
    label = "Novel" if win_rate >= 0.6 else ("Borderline" if win_rate >= 0.4 else "Incremental")

    console.print(Panel(
        f"Win-rate: [bold {color}]{pct}%[/]  ·  Assessment: [bold {color}]{label}[/]\n"
        f"[dim]Based on {len(scores)} Judge comparison(s)[/]",
        title="[bold]Novelty Assessment[/]",
        border_style=color,
    ))
    console.print()


def run_conclude_phase(
    investigation: Investigation,
    ai: AIClient,
    backend: ModelBackend | None = None,
) -> Investigation:
    """Run the conclusion phase.

    1. Judge scores the selected idea against reference papers.
    2. AI generates a structured research paper.
    3. Output saved to ~/.prism/investigations/<id>/paper.md.

    Args:
        investigation: The active investigation.
        ai: Orchestrator AI client.
        backend: Thinker/Judge backend. Falls back to configured backend if None.

    Returns:
        Updated investigation with conclusion and novelty_scores set.
    """
    from prism.config import get_backend

    if backend is None:
        backend = get_backend()

    print_phase_banner("conclusion")
    print_header(f"Conclusion: {investigation.title}")
    console.print()

    paper_count = len(investigation.papers)
    exp_count = len(investigation.experiments)
    completed = sum(1 for e in investigation.experiments if e.status == "complete")

    console.print(
        f"  Papers reviewed: [bold white]{paper_count}[/]   "
        f"Experiments: [bold white]{completed}/{exp_count} complete[/]"
    )
    if investigation.research_question:
        console.print(f"  Question: [italic white]{investigation.research_question}[/]")
    console.print()

    if not investigation.research_question and not investigation.papers:
        print_warning("No research question or papers. Consider running earlier phases first.")
        if not typer.confirm("Generate conclusion anyway?", default=False):
            return investigation

    # ── Step 1: Judge novelty scoring ─────────────────────────────────────────
    if investigation.papers:
        console.print("[bold cyan]Step 1 — Novelty Scoring[/]")
        console.print("[dim]Judge is comparing your idea against reference papers...[/]\n")

        with spinner("Running Judge comparisons..."):
            new_scores = _run_judge_scoring(investigation, backend)

        if new_scores:
            investigation.novelty_scores.extend(new_scores)
            win_rate = investigation.novelty_win_rate() or 0.0
            _display_novelty_report(new_scores, win_rate)
        else:
            print_info("No Judge scores produced (backend may be mock or unavailable).")
            console.print()

    # ── Step 2: Generate research summary ────────────────────────────────────
    console.print("[bold cyan]Step 2 — Research Summary[/]")
    print_info("Generating structured research summary...")

    with spinner("Writing research paper..."):
        try:
            conclusion_text = ai.generate_conclusion(investigation)
        except Exception as e:
            print_error(f"Failed to generate conclusion: {e}")
            conclusion_text = (
                f"# {investigation.title}\n\n"
                f"## Abstract\n\nThis investigation explored: {investigation.topic}\n\n"
                f"Research Question: {investigation.research_question or 'Not defined'}\n\n"
                f"## Summary\n\nPapers reviewed: {paper_count}  "
                f"Experiments completed: {completed}/{exp_count}\n\n"
                f"[Conclusion generation failed: {e}]"
            )

    console.print()
    console.rule("[bold green]Generated Research Summary[/]", style="green")
    console.print()
    console.print(Markdown(conclusion_text))
    console.print()
    console.rule(style="green")

    # ── Step 3: Save to disk ──────────────────────────────────────────────────
    inv_dir = PRISM_DIR / "investigations" / investigation.id
    inv_dir.mkdir(parents=True, exist_ok=True)
    paper_path = inv_dir / "paper.md"

    try:
        with open(paper_path, "w") as f:
            f.write(f"# {investigation.title}\n\n")
            if investigation.research_question:
                f.write(f"*Research Question:* {investigation.research_question}\n\n")
            if investigation.novelty_scores:
                win_rate = investigation.novelty_win_rate() or 0.0
                f.write(f"*Novelty Win-Rate:* {int(win_rate * 100)}% ({len(investigation.novelty_scores)} comparisons)\n\n")
            f.write(f"*Date:* {investigation.updated_at[:10]}\n\n---\n\n")
            f.write(conclusion_text)
        print_success(f"Saved to: {paper_path}")
    except Exception as e:
        print_warning(f"Could not save paper: {e}")

    if investigation.papers:
        if typer.confirm(f"Generate BibTeX for {paper_count} cited paper(s)?", default=True):
            bibtex = _generate_bibtex(investigation)
            bib_path = inv_dir / "references.bib"
            try:
                with open(bib_path, "w") as f:
                    f.write(bibtex)
                investigation.bibtex = bibtex
                print_success(f"BibTeX saved to: {bib_path}")
            except Exception as e:
                print_warning(f"Could not save BibTeX: {e}")

    investigation.conclusion = conclusion_text
    investigation.touch()

    console.print()
    print_success(f"Investigation complete. Files in: {inv_dir}")
    return investigation
