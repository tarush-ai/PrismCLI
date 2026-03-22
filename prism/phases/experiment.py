"""Experiment phase: plan and execute research experiments.

Prism owns: investigation state, phase transitions, experiment records.
Codex owns: code generation, refinement, and execution via `codex exec --full-auto`.

Failed experiments fail loudly and halt — no AI interpretation of broken runs.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import typer
from rich.markdown import Markdown
from rich.panel import Panel

from prism.models import Experiment, Investigation
from prism.display import (
    console,
    print_code,
    print_error,
    print_header,
    print_info,
    print_phase_banner,
    print_success,
    print_warning,
    spinner,
)
from prism.orchestrator import AIClient


# ── Execution helpers ──────────────────────────────────────────────────────────

def _run_python_direct(code: str, cwd: str | None = None) -> tuple[str, bool]:
    """Execute Python code in a subprocess.

    Returns:
        (output, success). On failure, output contains stderr. Does NOT
        call AI to interpret failures.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="prism_exp_"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout, True
        else:
            error_out = result.stdout
            if result.stderr:
                error_out += f"\n--- stderr ---\n{result.stderr}"
            return error_out, False
    except subprocess.TimeoutExpired:
        return "Execution timed out after 2 minutes.", False
    except Exception as e:
        return f"Execution failed: {e}", False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _build_handoff_prompt(exp: Experiment, investigation: Investigation) -> str:
    """Build the structured context prompt Prism hands to Codex."""
    papers_context = ""
    if investigation.papers:
        papers_context = "Reference papers:\n" + "\n".join(
            f"- {p.title} ({p.published[:4] if p.published else '?'})"
            for p in investigation.papers[:6]
        )

    return (
        f"# Prism Experiment Handoff\n\n"
        f"**Investigation:** {investigation.title}\n"
        f"**Research Question:** {investigation.research_question or 'Not yet defined'}\n\n"
        f"{papers_context}\n\n"
        f"---\n\n"
        f"**Experiment:** {exp.title}\n"
        f"**Description:** {exp.description}\n"
        f"**Hypothesis:** {exp.hypothesis or 'Not specified'}\n\n"
        f"**Starting code (refine and run this):**\n"
        f"```python\n{exp.code or '# No starting code — write from scratch'}\n```\n\n"
        f"Instructions:\n"
        f"1. Review or improve the code to properly test the hypothesis.\n"
        f"2. Run the code and capture results.\n"
        f"3. Print a clear summary of results to stdout.\n"
        f"4. If the code fails, fix it — do not stop on the first error.\n"
    )


def _run_via_codex(
    exp: Experiment,
    investigation: Investigation,
    ai: AIClient,
) -> tuple[str, bool]:
    """Hand off to Codex with live split-pane streaming.

    Left panel  → Prism investigation context (static).
    Right panel → Codex stdout/stderr lines as they arrive.

    Falls back to inline GPT-4o codegen + direct Python execution
    if Codex is not installed or produces no output.

    Returns:
        (output, success).
    """
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text

    handoff_prompt = _build_handoff_prompt(exp, investigation)

    # ── Build the static left-panel content ───────────────────────────────────
    left_lines = [
        f"[bold cyan]{investigation.title}[/]",
        "",
        f"[bold]Question[/]",
        (investigation.research_question or "[dim]not defined[/]")[:120],
        "",
        f"[bold]Experiment[/]",
        f"[white]{exp.title}[/]",
        "",
        f"[bold]Description[/]",
        exp.description[:180],
    ]
    if exp.hypothesis:
        left_lines += ["", f"[bold]Hypothesis[/]", exp.hypothesis[:180]]
    if investigation.papers:
        left_lines += ["", f"[bold]Papers ({len(investigation.papers)})[/]"]
        for p in investigation.papers[:5]:
            left_lines.append(f"  [dim]·[/] {p.title[:55]}")

    left_content = "\n".join(left_lines)

    # ── Layout ─────────────────────────────────────────────────────────────────
    layout = Layout()
    layout.split_row(
        Layout(name="context", ratio=2),
        Layout(name="codex",   ratio=3),
    )
    layout["context"].update(
        Panel(left_content, title="[bold cyan]Prism Context[/]", border_style="cyan", padding=(1, 2))
    )
    layout["codex"].update(
        Panel("[dim]Waiting for Codex...[/]", title="[bold magenta]Codex[/]", border_style="dim magenta")
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    timed_out = False
    codex_missing = False

    def _refresh_codex_panel(status: str = "") -> None:
        visible = stdout_lines[-45:] if stdout_lines else []
        body = Text()
        for line in visible:
            # Dim lines that look like boilerplate Codex UI chrome
            if line.startswith(("╭", "╰", "│", "┤", "├", "─")):
                body.append(line + "\n", style="dim")
            else:
                body.append(line + "\n")

        if stderr_lines:
            body.append("\n── stderr ──\n", style="bold red")
            for line in stderr_lines[-8:]:
                body.append(line + "\n", style="red")

        subtitle = status or f"[dim]{len(stdout_lines)} lines[/]"
        layout["codex"].update(
            Panel(body, title="[bold magenta]Codex[/]", subtitle=subtitle, border_style="magenta")
        )

    try:
        with Live(layout, refresh_per_second=8, screen=False, transient=False):
            for tag, line in ai.stream_codex_task(handoff_prompt, timeout=300):
                if tag == "stdout":
                    stdout_lines.append(line)
                    _refresh_codex_panel()
                elif tag == "stderr":
                    # Detect missing binary early
                    if "not found" in line.lower() or "No such file" in line:
                        codex_missing = True
                    stderr_lines.append(line)
                    _refresh_codex_panel()
                elif tag == "timeout":
                    timed_out = True
                    _refresh_codex_panel(status="[bold red]TIMED OUT[/]")
                # 'exit' — nothing extra needed

    except Exception as e:
        print_warning(f"Live display error: {e}")

    full_output = "\n".join(stdout_lines).strip()

    # ── Fallback: codegen via OpenAI + direct Python execution ────────────────
    if codex_missing or (not full_output and not timed_out):
        if codex_missing:
            print_warning("Codex not found. Falling back to inline code generation.")
        else:
            print_warning("Codex produced no output. Falling back to inline code generation.")

        with spinner("Generating experiment code via orchestrator..."):
            try:
                generated_code = ai.generate_experiment_code(
                    title=exp.title,
                    description=exp.description,
                    hypothesis=exp.hypothesis,
                    context=investigation.user_context or investigation.topic,
                )
            except Exception as e:
                return f"Code generation failed: {e}", False

        if generated_code:
            exp.code = generated_code
            print_code(generated_code, language="python", title="Generated Code")
            console.print("[bold cyan]Running generated code...[/]")
            output, success = _run_python_direct(generated_code)
            return output, success
        return "", False

    if timed_out:
        return "\n".join(stdout_lines), bool(stdout_lines)

    return full_output, bool(full_output)


# ── Display helpers ────────────────────────────────────────────────────────────

def _display_experiment_plan(experiments: list[dict]) -> None:
    from rich.table import Table

    table = Table(
        title="[bold magenta]Experiment Plan[/]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim magenta",
        show_lines=True,
    )
    table.add_column("#", width=4, justify="right", style="dim")
    table.add_column("Title", style="bold white", min_width=22)
    table.add_column("Description", style="white", min_width=30)
    table.add_column("Isolated", justify="center", width=9)

    for i, exp in enumerate(experiments, 1):
        isolated = exp.get("is_isolated", True)
        table.add_row(
            str(i),
            exp.get("title", "Untitled"),
            exp.get("description", "")[:90],
            "[green]yes[/]" if isolated else "[yellow]no[/]",
        )

    console.print(table)
    console.print()


def _fail_loudly(exp: Experiment, output: str) -> None:
    """Print a loud failure panel and mark the experiment failed."""
    exp.mark_failed(output)
    console.print()
    console.print(Panel(
        f"[bold red]{exp.title}[/]\n\n"
        f"[red]{output[:800]}[/]"
        + ("\n[dim]...(truncated)[/]" if len(output) > 800 else ""),
        title="[bold red]✗  EXPERIMENT FAILED[/]",
        border_style="red",
        padding=(1, 2),
    ))
    console.print()


# ── Core experiment runner ─────────────────────────────────────────────────────

def run_single_experiment(
    exp: Experiment,
    investigation: Investigation,
    ai: AIClient,
) -> Experiment:
    """Run a single experiment interactively.

    On failure: fails loudly with a red panel and halts — no AI interpretation.
    On success: AI interprets the result and suggests next steps.

    Returns:
        Updated experiment.
    """
    console.print()
    console.rule(f"[bold magenta]Experiment: {exp.title}[/]", style="magenta")
    console.print(f"[white]{exp.description}[/]")
    if exp.hypothesis:
        console.print(f"[dim]Hypothesis: {exp.hypothesis}[/]")
    console.print()

    console.print(
        "[bold cyan]Actions:[/]\n"
        "  [bold]c[/]  — hand off to Codex (generate + run)\n"
        "  [bold]p[/]  — run Python code directly\n"
        "  [bold]e[/]  — edit code, then run\n"
        "  [bold]v[/]  — view code only\n"
        "  [bold]s[/]  — skip this experiment\n"
    )

    code = exp.code

    while True:
        action = typer.prompt("Action").strip().lower()

        if action == "v":
            if code:
                print_code(code, language="python", title=exp.title)
            else:
                print_info("No code available.")
            console.print(
                "\n[dim cyan]Actions: [bold]c[/]odex / [bold]p[/]ython / "
                "[bold]e[/]dit / [bold]v[/]iew / [bold]s[/]kip[/]"
            )

        elif action == "e":
            if code:
                print_code(code, language="python", title="Current Code")
            else:
                print_info("No existing code.")
            console.print("[dim]Paste new Python code. Enter a line with only [bold]END[/] when done:[/]")
            lines: list[str] = []
            while True:
                try:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                except EOFError:
                    break
            if lines:
                code = "\n".join(lines)
                exp.code = code
                print_success("Code updated.")
            console.print(
                "\n[dim cyan]Actions: [bold]c[/]odex / [bold]p[/]ython / "
                "[bold]e[/]dit / [bold]v[/]iew / [bold]s[/]kip[/]"
            )

        elif action == "c":
            # ── Codex handoff ──────────────────────────────────────────────
            exp.mark_running()
            console.print()
            console.print(
                Panel(
                    f"[bold cyan]Handing off to Codex[/]\n"
                    f"[dim]Prism → Codex: experiment spec + context[/]",
                    border_style="cyan",
                    padding=(0, 2),
                )
            )
            console.print()

            output, codex_ok = _run_via_codex(exp, investigation, ai)

            if not codex_ok or not output:
                print_warning("Codex returned no output. Falling back to direct Python execution.")
                if code:
                    console.print("[dim]Running Python directly...[/]")
                    output, success = _run_python_direct(code)
                    if not success:
                        _fail_loudly(exp, output)
                        break
                    exp.mark_complete(output)
                    console.print("[bold green]Output:[/]")
                    console.print(f"[dim white]{output}[/]")
                else:
                    _fail_loudly(exp, "No code to run and Codex returned nothing.")
                    break
            else:
                exp.mark_complete(output)
                console.print("[bold cyan]Codex Output:[/]")
                console.print(Markdown(output) if output.startswith("#") else f"[dim white]{output}[/]")

            # AI interpretation on success only
            _interpret_success(exp, investigation, ai)
            break

        elif action == "p":
            if not code:
                print_warning("No code. Use 'e' to add code or 'c' to let Codex write it.")
                continue

            exp.mark_running()
            print_code(code, language="python", title="Executing")
            console.print("[bold cyan]Running...[/]")

            output, success = _run_python_direct(code)

            if not success:
                _fail_loudly(exp, output)
                break

            console.print("[bold green]Output:[/]")
            console.print(f"[dim white]{output}[/]")
            exp.mark_complete(output)

            # AI interpretation on success only
            _interpret_success(exp, investigation, ai)
            break

        elif action == "s":
            exp.mark_skipped()
            print_info(f"Skipped: {exp.title}")
            break

        else:
            print_warning("Enter c, p, e, v, or s.")

    return exp


def _interpret_success(
    exp: Experiment,
    investigation: Investigation,
    ai: AIClient,
) -> None:
    """Run AI interpretation after a successful experiment. Does nothing on failure."""
    if exp.status != "complete" or not exp.result or not investigation.research_question:
        return

    console.print()
    with spinner("Interpreting results..."):
        try:
            interpretation = ai.interpret_experiment_result(
                experiment_title=exp.title,
                code=exp.code,
                result=exp.result,
                research_question=investigation.research_question,
            )
        except Exception as e:
            print_warning(f"Interpretation failed: {e}")
            return

    exp.interpretation = interpretation
    console.print()
    console.rule("[dim cyan]AI Interpretation[/]", style="dim cyan")
    console.print(Markdown(interpretation))
    console.rule(style="dim cyan")


# ── Phase entry point ──────────────────────────────────────────────────────────

def run_experiment_phase(
    investigation: Investigation,
    ai: AIClient,
) -> Investigation:
    """Plan and run all experiments for an investigation.

    Experiments that fail halt the loop — the researcher must fix, skip, or abort.
    Skipped experiments are recorded but do not block phase advancement.

    Returns:
        Updated investigation with experiments added and phase set to 'conclusion'.
    """
    print_phase_banner("experiment")
    print_header(f"Experimentation: {investigation.title}")
    console.print()

    if not investigation.research_question:
        print_warning("No research question defined. Run 'prism question' first.")
        return investigation

    console.print(
        f"[bold cyan]Research Question:[/] "
        f"[italic white]{investigation.research_question}[/]\n"
    )

    # Generate experiment plan
    with spinner("Planning experiments..."):
        try:
            plan = ai.plan_experiments(
                question=investigation.research_question,
                context=investigation.user_context or investigation.topic,
            )
        except Exception as e:
            print_error(f"Failed to generate experiment plan: {e}")
            return investigation

    if not plan:
        print_warning("No experiments generated.")
        return investigation

    _display_experiment_plan(plan)

    if not typer.confirm(f"Proceed with {len(plan)} experiment(s)?", default=True):
        print_info("Experiment phase skipped.")
        return investigation

    # Create Experiment objects
    existing_titles = {e.title for e in investigation.experiments}
    new_experiments: list[Experiment] = []
    for exp_dict in plan:
        title = exp_dict.get("title", "Unnamed Experiment")
        if title in existing_titles:
            title = f"{title} (2)"
        exp = Experiment(
            id=str(uuid.uuid4()),
            investigation_id=investigation.id,
            title=title,
            description=exp_dict.get("description", ""),
            hypothesis=exp_dict.get("hypothesis", ""),
            code=exp_dict.get("code_template", ""),
            is_isolated=exp_dict.get("is_isolated", True),
            status="pending",
        )
        new_experiments.append(exp)

    investigation.experiments.extend(new_experiments)
    investigation.touch()

    from prism.db import save_experiment, update_experiment, update_investigation

    for exp in new_experiments:
        save_experiment(exp)

    # ── Run each experiment ────────────────────────────────────────────────────
    for exp in new_experiments:
        exp = run_single_experiment(exp, investigation, ai)
        update_experiment(exp)

        # Sync back into investigation list
        for i, e in enumerate(investigation.experiments):
            if e.id == exp.id:
                investigation.experiments[i] = exp
                break

        # Hard halt on failure — researcher must intervene
        if exp.status == "failed":
            console.print()
            halt = typer.confirm(
                "[bold red]Experiment failed.[/] Abort remaining experiments?",
                default=False,
            )
            if halt:
                print_warning("Halted by researcher after failed experiment.")
                update_investigation(investigation)
                raise typer.Exit(1)
            # else: continue with remaining experiments (researcher chose to proceed)

        # If non-isolated and succeeded, ask if plan should be re-evaluated
        if exp.status == "complete" and not exp.is_isolated and exp.interpretation:
            rethink = typer.confirm(
                "This experiment is non-isolated — re-evaluate remaining plan based on result?",
                default=False,
            )
            if rethink:
                remaining_idx = new_experiments.index(exp) + 1
                remaining = new_experiments[remaining_idx:]
                if remaining:
                    console.print("[dim]Re-planning remaining experiments...[/]")
                    # Just inform — full re-planning is a roadmap item
                    print_info(
                        f"{len(remaining)} experiment(s) remain. "
                        "Consider using 'prism chat' to discuss pivoting the plan."
                    )

    investigation.phase = "conclusion"
    investigation.touch()

    completed = sum(1 for e in new_experiments if e.status == "complete")
    failed = sum(1 for e in new_experiments if e.status == "failed")
    skipped = sum(1 for e in new_experiments if e.status == "skipped")

    print_success(
        f"Experiment phase complete: "
        f"{completed} succeeded, {failed} failed, {skipped} skipped."
    )
    return investigation


def run_named_experiment(
    exp_id: str,
    investigation: Investigation,
    ai: AIClient,
) -> None:
    """Run a single experiment by ID or prefix."""
    from prism.db import get_experiments, update_experiment, update_investigation

    experiments = get_experiments(investigation.id)
    if not experiments:
        print_error("No experiments found for this investigation.")
        raise typer.Exit(1)

    matched = [e for e in experiments if e.id == exp_id or e.id.startswith(exp_id)]
    if not matched:
        print_error(f"Experiment '{exp_id}' not found.")
        print_info("Use 'prism experiment list' to see all experiments.")
        raise typer.Exit(1)
    if len(matched) > 1:
        print_error(f"Ambiguous ID prefix '{exp_id}' matches {len(matched)} experiments.")
        raise typer.Exit(1)

    exp = matched[0]
    updated_exp = run_single_experiment(exp, investigation, ai)
    update_experiment(updated_exp)

    for i, e in enumerate(investigation.experiments):
        if e.id == updated_exp.id:
            investigation.experiments[i] = updated_exp
            break
    else:
        investigation.experiments.append(updated_exp)

    update_investigation(investigation)
