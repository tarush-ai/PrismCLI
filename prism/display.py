"""Rich-based display helpers for Prism CLI."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from prism.models import Experiment, Paper

# Shared console instance
console = Console()

# Phase colors
PHASE_COLORS = {
    "observation": "cyan",
    "research": "blue",
    "question": "yellow",
    "experiment": "magenta",
    "conclusion": "green",
}

PHASE_ICONS = {
    "observation": "🔭",
    "research": "📚",
    "question": "❓",
    "experiment": "🧪",
    "conclusion": "📝",
}

STATUS_COLORS = {
    "pending": "yellow",
    "running": "cyan",
    "complete": "green",
    "failed": "red",
}


def print_header(title: str) -> None:
    """Print a stylized panel header with Prism branding."""
    header_text = Text()
    header_text.append("✦ PRISM ", style="bold white")
    header_text.append("✦", style="bold cyan")
    header_text.append(f"  {title}", style="bold white")

    panel = Panel(
        header_text,
        border_style="cyan",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)


def print_phase_banner(phase: str) -> None:
    """Print a colorful phase indicator banner."""
    color = PHASE_COLORS.get(phase, "white")
    icon = PHASE_ICONS.get(phase, "•")
    phase_label = phase.upper()

    phases = ["observation", "research", "question", "experiment", "conclusion"]
    current_idx = phases.index(phase) if phase in phases else 0

    # Build progress bar
    progress_parts = []
    for i, p in enumerate(phases):
        p_color = PHASE_COLORS.get(p, "white")
        p_icon = PHASE_ICONS.get(p, "•")
        if i < current_idx:
            progress_parts.append(f"[dim {p_color}]{p_icon} {p.capitalize()}[/]")
        elif i == current_idx:
            progress_parts.append(f"[bold {p_color}]{p_icon} {p.capitalize()}[/]")
        else:
            progress_parts.append(f"[dim white]{p_icon} {p.capitalize()}[/]")

    progress_str = "  →  ".join(progress_parts)

    panel = Panel(
        f"{progress_str}",
        title=f"[bold {color}]{icon} Phase: {phase_label}[/]",
        border_style=color,
        padding=(0, 1),
    )
    console.print(panel)
    console.print()


def print_paper(paper: Paper, index: int | None = None) -> None:
    """Print a formatted paper card."""
    prefix = f"[bold cyan][{index}][/] " if index is not None else ""

    text = Text()
    text.append(f"{prefix}", style="")

    # Title
    console.print(f"{prefix}[bold white]{paper.title}[/]")

    # Authors and date
    authors_str = paper.short_authors(3)
    console.print(f"    [dim cyan]{authors_str}[/]  [dim]{paper.published}[/]")

    # Abstract (truncated)
    abstract = paper.abstract[:200].replace("\n", " ")
    if len(paper.abstract) > 200:
        abstract += "..."
    console.print(f"    [dim white]{abstract}[/]")

    # URL
    console.print(f"    [dim blue underline]{paper.url}[/]")
    console.print()


def print_investigation_table(investigations: list[dict]) -> None:
    """Print investigations as a rich Table."""
    if not investigations:
        print_info("No investigations found. Create one with: prism investigate new <topic>")
        return

    table = Table(
        title="[bold cyan]Investigations[/]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim cyan",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", style="bold white", min_width=20)
    table.add_column("Topic", style="white", min_width=15)
    table.add_column("Phase", justify="center", width=14)
    table.add_column("Created", style="dim", width=12)

    for i, inv in enumerate(investigations, 1):
        phase = inv.get("phase", "observation")
        phase_color = PHASE_COLORS.get(phase, "white")
        phase_icon = PHASE_ICONS.get(phase, "•")

        created = inv.get("created_at", "")
        if created and "T" in created:
            created = created.split("T")[0]

        short_id = inv.get("id", "")[:8]

        table.add_row(
            str(i),
            short_id,
            inv.get("title", ""),
            inv.get("topic", ""),
            f"[{phase_color}]{phase_icon} {phase.capitalize()}[/]",
            created,
        )

    console.print(table)


def print_experiment_status(exp: Experiment) -> None:
    """Print an experiment with colored status badge."""
    status_color = STATUS_COLORS.get(exp.status, "white")
    status_label = exp.status.upper()

    console.print(
        f"  [bold white]{exp.title}[/]  "
        f"[{status_color}]● {status_label}[/]"
    )
    console.print(f"  [dim]{exp.description[:120]}[/]")
    if exp.result:
        result_preview = exp.result[:150].replace("\n", " ")
        console.print(f"  [dim green]Result: {result_preview}[/]")
    console.print()


def print_success(msg: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/] {msg}")


def print_error(msg: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]✗[/] {msg}")


def print_info(msg: str) -> None:
    """Print an info message."""
    console.print(f"[bold cyan]ℹ[/] {msg}")


def print_warning(msg: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠[/] {msg}")


@contextmanager
def spinner(text: str) -> Generator[None, None, None]:
    """Context manager that shows a spinner while the block executes.

    Args:
        text: Text to display alongside the spinner.

    Yields:
        None
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(description=text, total=None)
        yield


def print_markdown_panel(content: str, title: str = "", border_style: str = "cyan") -> None:
    """Print content in a rich panel with optional markdown-like formatting."""
    from rich.markdown import Markdown

    md = Markdown(content)
    panel = Panel(md, title=f"[bold]{title}[/]" if title else "", border_style=border_style)
    console.print(panel)


def print_code(code: str, language: str = "python", title: str = "") -> None:
    """Print a code block with syntax highlighting."""
    from rich.syntax import Syntax

    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    if title:
        panel = Panel(syntax, title=f"[bold cyan]{title}[/]", border_style="dim cyan")
        console.print(panel)
    else:
        console.print(syntax)
