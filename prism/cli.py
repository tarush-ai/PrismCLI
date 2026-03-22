"""Main Typer CLI application for Prism."""

from __future__ import annotations

import uuid
from typing import Annotated, Optional

import typer
from rich.markdown import Markdown

from prism import __version__

app = typer.Typer(
    name="prism",
    help="Scientific research CLI — from observation to publication",
    add_completion=False,
    rich_markup_mode="rich",
)

investigate_app = typer.Typer(
    help="Manage investigations",
    add_completion=False,
    rich_markup_mode="rich",
)

experiment_app = typer.Typer(
    help="Manage experiments",
    add_completion=False,
    rich_markup_mode="rich",
)

app.add_typer(investigate_app, name="investigate")
app.add_typer(experiment_app, name="experiment")

from rich.console import Console
console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_api_key() -> str:
    from prism.config import get_api_key
    from prism.display import print_error, print_info

    key = get_api_key()
    if not key:
        print_error("No OpenAI API key configured.")
        print_info("Run 'prism init' to set up your API key.")
        raise typer.Exit(1)
    return key


def _get_ai_client() -> "AIClient":  # noqa: F821
    from prism.config import get_model
    from prism.orchestrator import AIClient

    api_key = _require_api_key()
    return AIClient(api_key=api_key, model=get_model())


def _require_active_investigation() -> "Investigation":  # noqa: F821
    from prism.config import get_active_investigation
    from prism.db import get_investigation, init_db
    from prism.display import print_error, print_info

    init_db()
    inv_id = get_active_investigation()
    if not inv_id:
        print_error("No active investigation set.")
        print_info(
            "Start one: prism investigate new\n"
            "Or select one: prism investigate use <id>"
        )
        raise typer.Exit(1)

    inv = get_investigation(inv_id)
    if inv is None:
        print_error(f"Active investigation '{inv_id}' not found in database.")
        print_info("Run 'prism investigate list' to see all investigations.")
        raise typer.Exit(1)

    return inv


# ── prism init ─────────────────────────────────────────────────────────────────

@app.command("init")
def cmd_init() -> None:
    """Interactive setup: configure API key, user profile, and model backend."""
    from prism.config import load_config, save_config
    from prism.display import print_error, print_header, print_info, print_success, print_warning

    print_header("Prism Setup")
    console.print(f"[dim]Version {__version__}[/]\n")

    config = load_config()

    # API key
    current_key = config.get("openai_api_key", "")
    masked = f"...{current_key[-6:]}" if len(current_key) > 6 else "(not set)"
    console.print(f"[bold cyan]OpenAI API Key[/] [dim](current: {masked})[/]")
    new_key = typer.prompt("Enter API key (Enter to keep current)", default="", hide_input=True).strip()
    if new_key:
        config["openai_api_key"] = new_key
        print_success("API key updated.")
    elif not current_key:
        print_error("No API key provided. You can also set OPENAI_API_KEY env var.")
        raise typer.Exit(1)

    # Model
    current_model = config.get("model", "gpt-4o")
    console.print(f"\n[bold cyan]Orchestrator Model[/] [dim](current: {current_model})[/]")
    new_model = typer.prompt("Model (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)", default=current_model).strip()
    config["model"] = new_model or current_model

    # User profile
    profile = config.get("user_profile", {"name": "", "research_areas": []})
    console.print(f"\n[bold cyan]Your Name[/] [dim](current: {profile.get('name') or 'not set'})[/]")
    name = typer.prompt("Name", default=profile.get("name", "")).strip()
    if name:
        profile["name"] = name

    current_areas = ", ".join(profile.get("research_areas", []))
    console.print(f"\n[bold cyan]Research Areas[/] [dim](current: {current_areas or 'none'})[/]")
    areas_input = typer.prompt(
        "Research areas, comma-separated (e.g. 'AI interpretability, NLP')",
        default=current_areas,
    ).strip()
    if areas_input:
        profile["research_areas"] = [a.strip() for a in areas_input.split(",") if a.strip()]
    config["user_profile"] = profile

    # Backend
    backend_cfg = config.get("backend", {})
    current_type = backend_cfg.get("type", "mock")
    console.print(f"\n[bold cyan]Model Backend[/] [dim](current: {current_type})[/]")
    console.print(
        "  [dim]mock[/]   — no local models, placeholder outputs\n"
        "  [dim]local[/]  — connect to a running vllm server (prism serve start)\n"
        "  [dim]remote[/] — connect to a hosted endpoint\n"
    )
    btype = typer.prompt("Backend type (mock/local/remote)", default=current_type).strip().lower()
    if btype in ("local", "remote"):
        current_url = backend_cfg.get("base_url", "http://localhost:8000/v1")
        url = typer.prompt("Backend base URL", default=current_url).strip()
        backend_cfg["base_url"] = url or current_url
        if btype == "remote":
            api_key = typer.prompt("Backend API key (Enter if none)", default="EMPTY", hide_input=True).strip()
            backend_cfg["api_key"] = api_key or "EMPTY"
    backend_cfg["type"] = btype if btype in ("mock", "local", "remote") else "mock"
    config["backend"] = backend_cfg

    save_config(config)

    # Test API connection
    console.print("\n[bold cyan]Testing OpenAI connection...[/]")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config["openai_api_key"])
        client.models.list()
        print_success("OpenAI connection OK.")
    except Exception as e:
        print_warning(f"OpenAI connection test failed: {e}")
        print_info("Configuration saved. Check your key if issues persist.")

    console.print()
    print_success("Prism configured. Start with:\n  prism investigate new")


# ── prism investigate new ──────────────────────────────────────────────────────

@investigate_app.command("new")
def investigate_new(
    topic_words: Annotated[
        Optional[list[str]],
        typer.Argument(help="Research topic (no quotes needed for multi-word topics)"),
    ] = None,
    title: Annotated[Optional[str], typer.Option("--title", "-t")] = None,
    public: Annotated[bool, typer.Option("--public", help="Make investigation public")] = False,
) -> None:
    """Start a new investigation. Runs as a conversation if no topic is given."""
    from prism.config import set_active_investigation
    from prism.db import create_investigation, init_db
    from prism.models import Investigation
    from prism.display import print_header, print_info, print_success

    init_db()
    print_header("New Investigation")

    # Conversational flow
    import re as _re

    def _clean_topic(t: str) -> str:
        t = _re.sub(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])?", "", t)
        t = _re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", t)
        return t.strip()

    if topic_words:
        topic = _clean_topic(" ".join(topic_words))
    else:
        console.print("[dim]What are you investigating?[/]")
        topic = _clean_topic(typer.prompt("Topic").strip())
        if not topic:
            from prism.display import print_error
            print_error("Topic cannot be empty.")
            raise typer.Exit(1)

    if not title:
        default_title = f"Investigation: {topic}"
        console.print(f"\n[dim]Working title:[/] [white]{default_title}[/]")
        title_input = typer.prompt("Title (Enter to keep)", default="").strip()
        title = title_input or default_title

    if not public:
        vis_choice = typer.prompt(
            "\nVisibility — [bold]private[/] (local only) or [bold]public[/] (Prism registry)",
            default="private",
        ).strip().lower()
        public = vis_choice == "public"

    inv = Investigation(
        id=str(uuid.uuid4()),
        title=title,
        topic=topic,
        phase="observation",
        visibility="public" if public else "private",
    )

    create_investigation(inv)
    set_active_investigation(inv.id)

    console.print()
    print_success(f"Created: [bold]{title}[/]")
    print_info(f"Topic: {topic}  ·  Visibility: {'public' if public else 'private'}")
    console.print()
    console.print(
        "[bold cyan]Run [bold white]prism go[/][/][bold cyan] to start the full research workflow, "
        "or [bold white]prism observe[/][/][bold cyan] for the first phase.[/]"
    )


# ── prism investigate list ─────────────────────────────────────────────────────

@investigate_app.command("list")
def investigate_list() -> None:
    """List all investigations."""
    from prism.db import init_db, list_investigations
    from prism.display import print_investigation_table

    init_db()
    print_investigation_table(list_investigations())


# ── prism investigate use ──────────────────────────────────────────────────────

@investigate_app.command("use")
def investigate_use(
    id_or_index: Annotated[str, typer.Argument(help="Investigation ID or list index")],
) -> None:
    """Set the active investigation."""
    from prism.config import set_active_investigation
    from prism.db import init_db, list_investigations
    from prism.display import print_error, print_info, print_success

    init_db()
    investigations = list_investigations()

    if not investigations:
        print_error("No investigations found.")
        raise typer.Exit(1)

    selected = None
    try:
        idx = int(id_or_index)
        if 1 <= idx <= len(investigations):
            selected = investigations[idx - 1]
        else:
            print_error(f"Index {idx} out of range (1–{len(investigations)}).")
            raise typer.Exit(1)
    except ValueError:
        matches = [i for i in investigations if i["id"] == id_or_index or i["id"].startswith(id_or_index)]
        if len(matches) == 1:
            selected = matches[0]
        elif len(matches) > 1:
            print_error(f"Ambiguous ID prefix. Be more specific.")
            raise typer.Exit(1)
        else:
            print_error(f"No investigation found with ID '{id_or_index}'.")
            raise typer.Exit(1)

    set_active_investigation(selected["id"])
    print_success(f"Active: [bold]{selected['title']}[/]")
    print_info(f"Phase: {selected['phase']}  ·  Topic: {selected['topic']}")


# ── prism investigate status ───────────────────────────────────────────────────

@investigate_app.command("status")
def investigate_status() -> None:
    """Show details of the current investigation."""
    from prism.db import get_experiments, init_db
    from prism.display import print_experiment_status, print_header, print_phase_banner

    init_db()
    inv = _require_active_investigation()

    print_header(inv.title)
    print_phase_banner(inv.phase)

    console.print(f"[bold cyan]ID:[/]       [dim]{inv.id}[/]")
    console.print(f"[bold cyan]Topic:[/]    {inv.topic}")
    console.print(f"[bold cyan]Visible:[/]  {inv.visibility}")
    console.print(f"[bold cyan]Created:[/]  {inv.created_at[:10]}")
    console.print()

    console.print(f"[bold cyan]Papers:[/] {len(inv.papers)}")
    for p in inv.papers[:5]:
        console.print(f"  • {p.title[:70]} [dim]({p.published[:4]})[/]")
    if len(inv.papers) > 5:
        console.print(f"  [dim]...and {len(inv.papers)-5} more[/]")
    console.print()

    if inv.thinker_ideas:
        selected = inv.selected_idea()
        console.print(f"[bold cyan]Thinker Ideas:[/] {len(inv.thinker_ideas)}")
        if selected:
            console.print(f"  [bold green]Selected:[/] {selected.title}")
        console.print()

    if inv.research_question:
        console.print(f"[bold cyan]Research Question:[/]")
        console.print(f"  [italic white]{inv.research_question}[/]")
        console.print()

    experiments = get_experiments(inv.id)
    console.print(f"[bold cyan]Experiments:[/] {len(experiments)}")
    for exp in experiments:
        print_experiment_status(exp)

    if inv.novelty_scores:
        win_rate = inv.novelty_win_rate()
        if win_rate is not None:
            console.print(f"[bold cyan]Novelty Win-Rate:[/] {int(win_rate * 100)}%")
        console.print()

    if inv.conclusion:
        console.print(f"[bold cyan]Conclusion:[/] [dim]generated ({len(inv.conclusion)} chars)[/]")


# ── prism investigate visibility ──────────────────────────────────────────────

@investigate_app.command("visibility")
def investigate_visibility(
    mode: Annotated[str, typer.Argument(help="public or private")],
) -> None:
    """Set investigation visibility to public or private."""
    from prism.db import init_db, update_investigation
    from prism.display import print_error, print_success

    init_db()
    inv = _require_active_investigation()
    mode = mode.strip().lower()

    if mode not in ("public", "private"):
        print_error("Mode must be 'public' or 'private'.")
        raise typer.Exit(1)

    inv.visibility = mode  # type: ignore[assignment]
    inv.touch()
    update_investigation(inv)
    print_success(f"Investigation visibility set to: {mode}")


# ── prism observe ──────────────────────────────────────────────────────────────

@app.command("observe")
def cmd_observe(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max papers to fetch")] = 0,
) -> None:
    """Phase 1: discover initial papers."""
    from prism.db import init_db, update_investigation
    from prism.display import print_info, print_warning
    from prism.phases.observe import run_observe_phase

    init_db()
    inv = _require_active_investigation()
    ai = _get_ai_client()

    if inv.phase != "observation" and inv.papers:
        print_warning(f"Investigation is at phase '{inv.phase}'. Running observe will add more papers.")
        if not typer.confirm("Continue?", default=True):
            raise typer.Exit(0)

    updated = run_observe_phase(inv, ai, limit=limit if limit > 0 else None)
    update_investigation(updated)
    print_info(f"Phase: {updated.phase}")


# ── prism research ─────────────────────────────────────────────────────────────

@app.command("research")
def cmd_research() -> None:
    """Phase 2: deeper literature synthesis."""
    from prism.db import init_db, update_investigation
    from prism.display import print_info, print_warning
    from prism.phases.research import run_research_phase

    init_db()
    inv = _require_active_investigation()
    ai = _get_ai_client()

    if inv.phase not in ("research", "observation"):
        print_warning(f"Investigation is at phase '{inv.phase}'. Expected 'research'.")
        if not typer.confirm("Run anyway?", default=True):
            raise typer.Exit(0)

    updated = run_research_phase(inv, ai)
    update_investigation(updated)
    print_info(f"Phase: {updated.phase}")


# ── prism question ─────────────────────────────────────────────────────────────

@app.command("question")
def cmd_question() -> None:
    """Phase 3: Thinker-driven ideation and research question definition."""
    from prism.config import get_backend
    from prism.db import init_db, update_investigation
    from prism.display import print_info, print_warning
    from prism.phases.question import run_question_phase

    init_db()
    inv = _require_active_investigation()
    ai = _get_ai_client()
    backend = get_backend()

    if inv.phase not in ("question", "research"):
        print_warning(f"Investigation is at phase '{inv.phase}'. Expected 'question'.")
        if not typer.confirm("Run anyway?", default=True):
            raise typer.Exit(0)

    updated = run_question_phase(inv, ai, backend=backend)
    update_investigation(updated)
    print_info(f"Phase: {updated.phase}")


# ── prism idea ─────────────────────────────────────────────────────────────────

idea_app = typer.Typer(help="Inspect Thinker-proposed ideas", add_completion=False)
app.add_typer(idea_app, name="idea")


@idea_app.command("list")
def idea_list() -> None:
    """List all Thinker-proposed ideas for the active investigation."""
    from prism.display import print_info

    inv = _require_active_investigation()
    if not inv.thinker_ideas:
        print_info("No Thinker ideas yet. Run 'prism question' first.")
        return

    console.print(f"[bold cyan]Thinker Ideas — {inv.title}[/]\n")
    for i, idea in enumerate(inv.thinker_ideas, 1):
        selected_marker = " [bold green]✓[/]" if idea.selected else ""
        console.print(f"  [dim]{idea.id[:8]}[/]  [bold white]{idea.title}[/]{selected_marker}")
        console.print(f"  [dim]from: {idea.source_paper_id[:8]}[/]")
        console.print()


@idea_app.command("trace")
def idea_trace(
    idea_id: Annotated[str, typer.Argument(help="Idea ID or prefix")],
) -> None:
    """Inspect the Thinker's chain-of-thought for a proposed idea."""
    from prism.display import print_error, print_info

    inv = _require_active_investigation()
    matches = [i for i in inv.thinker_ideas if i.id == idea_id or i.id.startswith(idea_id)]

    if not matches:
        print_error(f"Idea '{idea_id}' not found.")
        raise typer.Exit(1)
    if len(matches) > 1:
        print_error("Ambiguous ID prefix.")
        raise typer.Exit(1)

    idea = matches[0]
    console.print(f"[bold cyan]Idea:[/] {idea.title}\n")

    if not idea.thinking_trace or idea.thinking_trace.startswith("[Mock"):
        print_info("No thinking trace available for this idea.")
        return

    console.rule("[dim]Thinker Chain-of-Thought[/]", style="dim")
    console.print(Markdown(idea.thinking_trace))
    console.rule(style="dim")


@idea_app.command("select")
def idea_select(
    idea_id: Annotated[str, typer.Argument(help="Idea ID or prefix to select")],
) -> None:
    """Manually select a Thinker idea as the working research question."""
    from prism.db import init_db, update_investigation
    from prism.display import print_error, print_success

    init_db()
    inv = _require_active_investigation()
    matches = [i for i in inv.thinker_ideas if i.id == idea_id or i.id.startswith(idea_id)]

    if not matches:
        print_error(f"Idea '{idea_id}' not found.")
        raise typer.Exit(1)
    if len(matches) > 1:
        print_error("Ambiguous ID prefix.")
        raise typer.Exit(1)

    for i in inv.thinker_ideas:
        i.selected = (i.id == matches[0].id)

    inv.research_question = matches[0].title
    inv.touch()
    update_investigation(inv)
    print_success(f"Selected: {matches[0].title}")


# ── prism experiment plan ──────────────────────────────────────────────────────

@experiment_app.command("plan")
def experiment_plan() -> None:
    """Phase 4: plan and run experiments."""
    from prism.db import init_db, update_investigation
    from prism.display import print_info, print_warning
    from prism.phases.experiment import run_experiment_phase

    init_db()
    inv = _require_active_investigation()
    ai = _get_ai_client()

    if inv.phase not in ("experiment", "question"):
        print_warning(f"Investigation is at phase '{inv.phase}'. Expected 'experiment'.")
        if not typer.confirm("Run anyway?", default=True):
            raise typer.Exit(0)

    updated = run_experiment_phase(inv, ai)
    update_investigation(updated)
    print_info(f"Phase: {updated.phase}")


@experiment_app.command("run")
def experiment_run(
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID or prefix")],
) -> None:
    """Run a specific experiment by ID."""
    from prism.db import init_db
    from prism.phases.experiment import run_named_experiment

    init_db()
    inv = _require_active_investigation()
    ai = _get_ai_client()
    run_named_experiment(experiment_id, inv, ai)


@experiment_app.command("list")
def experiment_list() -> None:
    """List all experiments for the active investigation."""
    from prism.db import get_experiments, init_db
    from prism.display import print_experiment_status, print_info

    init_db()
    inv = _require_active_investigation()
    experiments = get_experiments(inv.id)

    if not experiments:
        print_info("No experiments found. Run 'prism experiment plan'.")
        return

    console.print(f"[bold cyan]Experiments — {inv.title}[/] [dim]({len(experiments)} total)[/]\n")
    for exp in experiments:
        console.print(f"  [dim]{exp.id[:8]}[/]  ", end="")
        print_experiment_status(exp)


# ── prism conclude ─────────────────────────────────────────────────────────────

@app.command("conclude")
def cmd_conclude() -> None:
    """Phase 5: Judge novelty scoring + generate research paper."""
    from prism.config import get_backend
    from prism.db import init_db, update_investigation
    from prism.display import print_info, print_warning
    from prism.phases.conclude import run_conclude_phase

    init_db()
    inv = _require_active_investigation()
    ai = _get_ai_client()
    backend = get_backend()

    if inv.phase not in ("conclusion", "experiment"):
        print_warning(f"Investigation is at phase '{inv.phase}'. Expected 'conclusion'.")
        if not typer.confirm("Generate conclusion anyway?", default=True):
            raise typer.Exit(0)

    updated = run_conclude_phase(inv, ai, backend=backend)
    update_investigation(updated)
    print_info(f"Phase: {updated.phase}")


# ── prism chat ─────────────────────────────────────────────────────────────────

@app.command("chat")
def cmd_chat() -> None:
    """Open-ended AI chat within the current investigation context. Ctrl+C to exit."""
    from prism.display import print_header, print_info, print_warning

    inv = _require_active_investigation()
    ai = _get_ai_client()

    print_header(f"Chat: {inv.title}")
    print_info("Chatting within investigation context. Press Ctrl+C or type 'exit' to quit.\n")

    system_parts = [
        "You are a research assistant helping with the following investigation.",
        f"Title: {inv.title}",
        f"Topic: {inv.topic}",
        f"Current phase: {inv.phase}",
    ]
    if inv.research_question:
        system_parts.append(f"Research question: {inv.research_question}")
    if inv.user_context:
        system_parts.append(f"Background context:\n{inv.user_context[:600]}")
    if inv.papers:
        titles = "\n".join(f"- {p.title}" for p in inv.papers[:6])
        system_parts.append(f"Papers reviewed:\n{titles}")

    system_prompt = "\n\n".join(system_parts)
    conversation: list[dict] = []

    try:
        while True:
            user_input = typer.prompt("\nYou").strip()
            if not user_input:
                continue
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                break

            conversation.append({"role": "user", "content": user_input})

            console.print("\n[bold cyan]Prism:[/] ", end="")
            response_parts: list[str] = []
            try:
                for token in ai.stream_chat(conversation, system=system_prompt):
                    console.print(token, end="")
                    response_parts.append(token)
                console.print()
            except Exception as e:
                print_warning(f"\nStreaming failed: {e}. Retrying...")
                try:
                    full = ai.chat(conversation, system=system_prompt)
                    console.print(full)
                    response_parts = [full]
                except Exception as e2:
                    print_warning(f"Chat failed: {e2}")
                    conversation.pop()
                    continue

            conversation.append({"role": "assistant", "content": "".join(response_parts)})
            if len(conversation) > 20:
                conversation = conversation[-18:]

    except (KeyboardInterrupt, EOFError):
        console.print()
        print_info("Chat ended.")


# ── prism go ───────────────────────────────────────────────────────────────────

@app.command("go")
def cmd_go() -> None:
    """Autopilot: run all remaining phases from current phase.

    This is the primary interface. Discrete commands (observe, research, etc.)
    are researcher escape hatches for when you need to re-run a single phase.
    """
    from prism.config import get_backend
    from prism.db import init_db, update_investigation
    from prism.display import print_header, print_info, print_success, print_warning

    init_db()
    inv = _require_active_investigation()
    ai = _get_ai_client()
    backend = get_backend()

    phases_order = ["observation", "research", "question", "experiment", "conclusion"]
    current_idx = phases_order.index(inv.phase)
    remaining = phases_order[current_idx:]

    print_header(f"Autopilot: {inv.title}")
    print_info(f"Running: {' → '.join(p.capitalize() for p in remaining)}\n")

    for phase in remaining:
        print_info(f"\nStarting: [bold]{phase.capitalize()}[/]")
        try:
            if phase == "observation":
                from prism.phases.observe import run_observe_phase
                inv = run_observe_phase(inv, ai)

            elif phase == "research":
                from prism.phases.research import run_research_phase
                inv = run_research_phase(inv, ai)

            elif phase == "question":
                from prism.phases.question import run_question_phase
                inv = run_question_phase(inv, ai, backend=backend)

            elif phase == "experiment":
                from prism.phases.experiment import run_experiment_phase
                inv = run_experiment_phase(inv, ai)

            elif phase == "conclusion":
                from prism.phases.conclude import run_conclude_phase
                inv = run_conclude_phase(inv, ai, backend=backend)

            update_investigation(inv)
            print_success(f"Phase [bold]{phase}[/] complete.")

        except SystemExit:
            raise
        except KeyboardInterrupt:
            update_investigation(inv)
            print_warning(f"\nInterrupted at phase '{phase}'. Progress saved.")
            raise typer.Exit(0)
        except Exception as e:
            update_investigation(inv)
            print_warning(f"Phase '{phase}' error: {e}")
            if not typer.confirm("Continue to next phase?", default=True):
                raise typer.Exit(1)

    print_success("All phases complete.")


# ── prism serve ────────────────────────────────────────────────────────────────

serve_app = typer.Typer(help="Manage local model server (vllm)", add_completion=False)
app.add_typer(serve_app, name="serve")


@serve_app.command("status")
def serve_status() -> None:
    """Check if the local model server is reachable."""
    from prism.config import get_backend_config
    from prism.display import print_error, print_success

    cfg = get_backend_config()
    url = cfg.get("base_url", "http://localhost:8000/v1")

    import httpx
    try:
        with httpx.Client(timeout=3) as client:
            resp = client.get(f"{url}/models")
            if resp.status_code == 200:
                print_success(f"Server reachable at {url}")
                data = resp.json()
                for m in data.get("data", []):
                    console.print(f"  • {m.get('id', '?')}")
            else:
                print_error(f"Server returned {resp.status_code}")
    except Exception as e:
        print_error(f"Cannot reach server at {url}: {e}")
        console.print("[dim]Start it with: prism serve start[/]")


@serve_app.command("start")
def serve_start(
    thinker_only: Annotated[bool, typer.Option("--thinker-only")] = False,
    judge_only: Annotated[bool, typer.Option("--judge-only")] = False,
) -> None:
    """Print instructions for starting the local vllm model server."""
    from prism.config import get_backend_config
    from prism.display import print_info

    cfg = get_backend_config()
    thinker = cfg.get("thinker_model", "OpenMOSS-Team/SciThinker-30B")
    judge = cfg.get("judge_model", "OpenMOSS-Team/SciJudge-30B")
    port = cfg.get("base_url", "http://localhost:8000/v1").split(":")[-1].split("/")[0]

    print_info("To start the local model server, run in a separate terminal:")
    console.print()
    if not judge_only:
        console.print(
            f"[bold cyan]# Thinker[/]\n"
            f"vllm serve {thinker} --port {port} --served-model-name {thinker}\n"
        )
    if not thinker_only:
        console.print(
            f"[bold cyan]# Judge[/] (use a different port if running both)\n"
            f"vllm serve {judge} --port {port} --served-model-name {judge}\n"
        )
    console.print(
        "[dim]Then update backend.base_url in your config if needed, "
        "and run: prism config set backend.type local[/]"
    )


# ── prism config ───────────────────────────────────────────────────────────────

config_app = typer.Typer(help="View and edit configuration", add_completion=False)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """Show the current configuration."""
    import json
    from prism.config import load_config
    config = load_config()
    # Mask API keys
    if config.get("openai_api_key"):
        key = config["openai_api_key"]
        config["openai_api_key"] = f"...{key[-6:]}" if len(key) > 6 else "***"
    if config.get("backend", {}).get("api_key") not in ("", "EMPTY", None):
        config["backend"]["api_key"] = "***"
    console.print_json(json.dumps(config, indent=2))


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Dotted key path, e.g. observation.paper_limit")],
    value: Annotated[str, typer.Argument(help="Value to set")],
) -> None:
    """Set a configuration value by dotted path."""
    from prism.config import load_config, save_config
    from prism.display import print_error, print_success

    config = load_config()
    parts = key.split(".")

    # Navigate to the parent
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]

    # Parse value: try int, float, bool, then string
    parsed: object = value
    if value.lower() in ("true", "yes"):
        parsed = True
    elif value.lower() in ("false", "no"):
        parsed = False
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value

    target[parts[-1]] = parsed
    save_config(config)
    print_success(f"Set {key} = {parsed!r}")


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="Dotted key path")],
) -> None:
    """Get a configuration value."""
    from prism.config import load_config
    from prism.display import print_error

    config = load_config()
    parts = key.split(".")
    target: object = config
    for part in parts:
        if isinstance(target, dict) and part in target:
            target = target[part]
        else:
            print_error(f"Key '{key}' not found.")
            raise typer.Exit(1)
    console.print(str(target))


# ── Version callback ───────────────────────────────────────────────────────────

def _version_callback(value: bool) -> None:
    if value:
        console.print(f"prism {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """Prism — scientific research CLI, from observation to publication."""
    pass


if __name__ == "__main__":
    app()
