"""Question definition phase: Thinker-driven ideation + researcher selection."""

from __future__ import annotations

import uuid

import typer
from rich.markdown import Markdown

from prism.taste.base import ModelBackend
from prism.models import Investigation, ProposedIdea
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


def _display_idea(idea: ProposedIdea, index: int, source_title: str = "") -> None:
    """Render a single ProposedIdea as a numbered card."""
    console.print(f"  [bold cyan][{index}][/]  [bold white]{idea.title}[/]")
    if source_title:
        console.print(f"       [dim]spawned from: {source_title[:60]}[/]")
    console.print()
    abstract_md = Markdown(idea.abstract)
    console.print("       ", end="")
    console.print(abstract_md)
    console.print()


def run_question_phase(
    investigation: Investigation,
    ai: AIClient,
    backend: ModelBackend | None = None,
) -> Investigation:
    """Run the question definition phase.

    1. AI gap analysis surfaces what is genuinely missing in the literature.
    2. Thinker explores each reference paper and proposes follow-up ideas.
    3. Researcher selects an idea or writes their own.
    4. Optional iterative refinement with orchestrator (GPT-4o).

    Args:
        investigation: The active investigation.
        ai: Orchestrator AI client (GPT-4o).
        backend: Thinker/Judge backend. Falls back to mock if None.

    Returns:
        Updated investigation with research_question and thinker_ideas set.
    """
    from prism.config import get_backend, get_thinker_config

    if backend is None:
        backend = get_backend()

    thinker_cfg = get_thinker_config()
    num_ideas = thinker_cfg.get("num_ideas_per_paper", 3)
    use_future = thinker_cfg.get("use_future_explorations", True)

    print_phase_banner("question")
    print_header(f"Research Question: {investigation.title}")
    console.print()

    if not investigation.papers:
        print_warning("No papers in this investigation. Run 'prism observe' and 'prism research' first.")
        return investigation

    # ── Step 1: Gap analysis ──────────────────────────────────────────────────
    console.print("[bold cyan]Step 1 — Gap Analysis[/]")
    console.print(
        f"[dim]Analysing {len(investigation.papers)} paper(s) for open problems and unresolved questions...[/]\n"
    )

    gap_text = ""
    with spinner("Identifying gaps in the literature..."):
        try:
            gap_text = ai.gap_analysis(investigation.papers, investigation.topic)
        except Exception as e:
            print_warning(f"Gap analysis failed: {e}")
            gap_text = ""

    if gap_text:
        console.rule("[dim cyan]Literature Gaps[/]", style="dim cyan")
        console.print(Markdown(gap_text))
        console.rule(style="dim cyan")
        console.print()

    # ── Step 2: Thinker ideation ──────────────────────────────────────────────
    console.print("[bold cyan]Step 2 — Thinker Ideation[/]")

    papers_to_explore = investigation.papers[:5]   # cap at 5 to stay reasonable
    console.print(
        f"[dim]Exploring {len(papers_to_explore)} paper(s) with SciThinker "
        f"({num_ideas} ideas each)...[/]\n"
    )

    all_new_ideas: list[ProposedIdea] = []
    paper_title_map: dict[str, str] = {p.id: p.title for p in investigation.papers}

    for paper in papers_to_explore:
        with spinner(f"Thinker exploring: {paper.title[:55]}..."):
            try:
                ideas = backend.explore(paper, num_ideas=num_ideas, use_future_explorations=use_future)
                all_new_ideas.extend(ideas)
            except Exception as e:
                print_warning(f"Thinker failed on '{paper.title[:40]}': {e}")

    if not all_new_ideas:
        print_warning("Thinker produced no ideas. Falling back to orchestrator proposals.")
        with spinner("Generating proposals via orchestrator..."):
            try:
                fallback_text = ai.generate_research_question(
                    background=investigation.user_context or investigation.topic,
                    papers=investigation.papers,
                )
                # Wrap as a single ProposedIdea so the rest of the flow is uniform
                all_new_ideas = [
                    ProposedIdea(
                        id=str(uuid.uuid4()),
                        title="Orchestrator Proposal",
                        abstract=fallback_text,
                        source_paper_id=investigation.papers[0].id if investigation.papers else "",
                    )
                ]
            except Exception as e:
                print_error(f"Orchestrator proposal also failed: {e}")
                return investigation

    # Persist all Thinker ideas on the investigation
    existing_idea_titles = {i.title for i in investigation.thinker_ideas}
    for idea in all_new_ideas:
        if idea.title not in existing_idea_titles:
            investigation.thinker_ideas.append(idea)

    # ── Step 3: Present ideas ─────────────────────────────────────────────────
    console.print()
    console.print(
        f"[bold cyan]Proposed Ideas[/] [dim]({len(all_new_ideas)} total from Thinker)[/]"
    )
    console.rule(style="dim cyan")
    console.print()

    for i, idea in enumerate(all_new_ideas, 1):
        source_title = paper_title_map.get(idea.source_paper_id, "")
        _display_idea(idea, i, source_title)

    console.rule(style="dim cyan")
    console.print()

    # ── Step 4: Selection ─────────────────────────────────────────────────────
    console.print(
        "[bold cyan]Select an idea:[/]\n"
        f"  [bold]1–{len(all_new_ideas)}[/]  select a Thinker-proposed idea\n"
        "  [bold]o[/]       write your own research question\n"
    )

    chosen_question: str | None = None
    chosen_idea: ProposedIdea | None = None

    while chosen_question is None:
        choice = typer.prompt("Choice").strip().lower()

        if choice == "o":
            custom = typer.prompt("Enter your research question").strip()
            if custom:
                chosen_question = custom
            else:
                print_warning("Question cannot be empty.")
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(all_new_ideas):
                    chosen_idea = all_new_ideas[idx - 1]
                    chosen_question = chosen_idea.title
                else:
                    print_warning(f"Please enter 1–{len(all_new_ideas)} or 'o'.")
            except ValueError:
                print_warning("Invalid input.")

    # Mark selection
    if chosen_idea:
        for idea in investigation.thinker_ideas:
            idea.selected = (idea.id == chosen_idea.id)

    console.print()
    console.print("[bold cyan]Selected:[/]")
    console.print(f"[bold white italic]\"{chosen_question}\"[/]")
    console.print()

    # ── Step 5: Refinement ────────────────────────────────────────────────────
    while True:
        refine = typer.confirm("Refine with AI?", default=False)
        if not refine:
            break

        feedback = typer.prompt("How to refine (e.g. 'more specific', 'focus on efficiency')").strip()
        if not feedback:
            break

        with spinner("Refining..."):
            try:
                refined_text = ai.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Current research question: {chosen_question}\n"
                                f"Feedback: {feedback}\n"
                                f"Background context: {(investigation.user_context or investigation.topic)[:500]}\n\n"
                                f"Propose 3 refined versions that sharpen this into a concrete, "
                                f"testable research question. Number them. Each on its own line. "
                                f"No preamble."
                            ),
                        }
                    ],
                    system=(
                        "You are a research methodologist. Return exactly 3 concise, "
                        "testable research questions. No explanations — just the questions numbered."
                    ),
                )
            except Exception as e:
                print_warning(f"Refinement failed: {e}")
                break

        import re
        lines = [l.strip() for l in refined_text.strip().splitlines() if l.strip()]
        refined_options = []
        for line in lines:
            clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if clean and len(clean) > 15:
                refined_options.append(clean)
        refined_options = refined_options[:3]

        if not refined_options:
            print_warning("Refinement returned no parseable options.")
            break

        console.print()
        for i, q in enumerate(refined_options, 1):
            console.print(f"  [bold cyan][{i}][/] {q}")
        console.print()

        while True:
            pick = typer.prompt(
                f"Select refined question (1–{len(refined_options)}), or 'k' to keep current"
            ).strip().lower()
            if pick == "k":
                break
            try:
                idx = int(pick)
                if 1 <= idx <= len(refined_options):
                    chosen_question = refined_options[idx - 1]
                    # Deselect previous idea since question changed
                    for idea in investigation.thinker_ideas:
                        idea.selected = False
                    console.print(f"\n[bold green]Updated:[/] {chosen_question}\n")
                    break
                else:
                    print_warning(f"Enter 1–{len(refined_options)} or 'k'.")
            except ValueError:
                print_warning("Invalid input.")

    # ── Finalise ──────────────────────────────────────────────────────────────
    console.print()
    console.print("[bold cyan]Final Research Question:[/]")
    console.print(f"[bold white italic]\"{chosen_question}\"[/]")
    console.print()

    investigation.research_question = chosen_question
    investigation.phase = "experiment"
    investigation.touch()

    print_success("Research question locked. Ready for experimentation.")
    return investigation
