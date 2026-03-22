"""OpenAI AI client wrapper for Prism."""

from __future__ import annotations

import subprocess
import sys
from typing import Iterator

from openai import OpenAI, APIError, AuthenticationError, RateLimitError

from prism.models import Investigation, Paper


class AIClient:
    """Wrapper around the OpenAI client for Prism operations."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        """Initialize the AI client.

        Args:
            api_key: OpenAI API key.
            model: Model name to use (default: gpt-4o).
        """
        self.api_key = api_key
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with "role" and "content".
            system: Optional system prompt.

        Returns:
            Response text string.
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,  # type: ignore[arg-type]
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except AuthenticationError:
            raise RuntimeError(
                "Invalid OpenAI API key. Run 'prism init' to reconfigure."
            )
        except RateLimitError:
            raise RuntimeError(
                "OpenAI rate limit reached. Please wait a moment and try again."
            )
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def stream_chat(
        self, messages: list[dict], system: str | None = None
    ) -> Iterator[str]:
        """Stream a chat completion, yielding tokens as they arrive.

        Args:
            messages: List of message dicts with "role" and "content".
            system: Optional system prompt.

        Yields:
            Text tokens from the response.
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,  # type: ignore[arg-type]
                temperature=0.7,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except AuthenticationError:
            raise RuntimeError(
                "Invalid OpenAI API key. Run 'prism init' to reconfigure."
            )
        except RateLimitError:
            raise RuntimeError(
                "OpenAI rate limit reached. Please wait a moment and try again."
            )
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def stream_codex_task(
        self,
        task: str,
        cwd: str | None = None,
        timeout: int = 300,
    ):
        """Stream a Codex task live, yielding (tag, line) tuples as they arrive.

        Uses Popen + background threads so stdout and stderr are both visible
        in real time instead of being swallowed until the process exits.

        Args:
            task: The task prompt for `codex exec --full-auto`.
            cwd: Working directory (defaults to home).
            timeout: Hard kill timeout in seconds.

        Yields:
            ('stdout', line) or ('stderr', line) tuples.
            ('timeout', msg) if the process is killed for taking too long.
            ('exit', returncode_str) as the final item.
        """
        import queue
        import threading
        import time
        from pathlib import Path

        try:
            proc = subprocess.Popen(
                [
                    "codex", "exec",
                    "--full-auto",
                    "--skip-git-repo-check",
                    task,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=cwd or str(Path.home()),
            )
        except FileNotFoundError:
            yield ("stderr", "codex CLI not found — install with: npm install -g @openai/codex")
            yield ("exit", "1")
            return

        line_queue: queue.Queue = queue.Queue()

        def _reader(stream, tag: str) -> None:
            try:
                for raw in iter(stream.readline, ""):
                    line_queue.put((tag, raw.rstrip()))
            finally:
                line_queue.put((tag, None))  # sentinel

        t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
        t_out.start()
        t_err.start()

        deadline = time.monotonic() + timeout
        done = 0

        while done < 2:
            if time.monotonic() > deadline:
                proc.kill()
                yield ("timeout", f"Codex killed after {timeout}s timeout")
                break
            try:
                tag, line = line_queue.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None:
                    # drain anything left
                    while not line_queue.empty():
                        try:
                            tag, line = line_queue.get_nowait()
                            if line is not None:
                                yield (tag, line)
                        except queue.Empty:
                            break
                    break
                continue
            if line is None:
                done += 1
            else:
                yield (tag, line)

        proc.wait()
        yield ("exit", str(proc.returncode))

    def run_codex_task(self, task: str, cwd: str | None = None, timeout: int = 300) -> str:
        """Blocking wrapper around stream_codex_task — collects all output.

        Prefer stream_codex_task directly when you want live display.
        This exists for callers that just need the final text.
        """
        from prism.display import print_warning

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        for tag, line in self.stream_codex_task(task, cwd=cwd, timeout=timeout):
            if tag == "stdout":
                stdout_lines.append(line)
            elif tag == "stderr":
                stderr_lines.append(line)
            elif tag == "timeout":
                print_warning(line)
            # 'exit' sentinel — ignore

        output = "\n".join(stdout_lines).strip()
        if not output and stderr_lines:
            print_warning("Codex produced no stdout. Stderr:\n" + "\n".join(stderr_lines[:10]))
        return output

    def generate_experiment_code(
        self,
        title: str,
        description: str,
        hypothesis: str,
        context: str,
    ) -> str:
        """Generate self-contained Python experiment code via the orchestrator.

        Used as a fallback when Codex is unavailable or produces no output.

        Returns:
            Runnable Python source string.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Write complete, self-contained, runnable Python code for this experiment.\n\n"
                    f"Experiment: {title}\n"
                    f"Description: {description}\n"
                    f"Hypothesis: {hypothesis or 'not specified'}\n"
                    f"Context: {context[:600]}\n\n"
                    f"Requirements:\n"
                    f"- Use only stdlib + numpy/matplotlib/sklearn/pandas (standard packages)\n"
                    f"- Print a clear results summary to stdout\n"
                    f"- Must run without arguments\n"
                    f"- No TODOs, no placeholder values\n\n"
                    f"Return ONLY the Python code inside a ```python block."
                ),
            }
        ]
        response = self.chat(
            messages,
            system="You are an expert computational scientist. Return only runnable Python code.",
        )
        # Extract code from markdown fence
        import re
        m = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def gap_analysis(self, papers: list[Paper], topic: str) -> str:
        """Identify explicit gaps, open problems, and future work from a set of papers.

        This drives Phase 3 question proposals. Unlike theme summarisation, it
        specifically looks for what is *missing* or *unresolved* in the literature.

        Args:
            papers: Papers already selected by the researcher.
            topic: The investigation topic for framing.

        Returns:
            Structured gap analysis: list of open problems + why each matters.
        """
        summaries = "\n\n".join(
            f"[{i+1}] **{p.title}** ({p.published[:4] if p.published else '?'})\n"
            f"Authors: {p.short_authors(2)}\n"
            f"Abstract: {p.abstract[:600]}"
            + (f"\nFuture Work: {p.future_explorations[:300]}" if p.future_explorations else "")
            for i, p in enumerate(papers[:12])
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Papers:\n{summaries}\n\n"
                    f"Perform a rigorous gap analysis. Do NOT summarise what these papers do. "
                    f"Instead, identify:\n"
                    f"1. **Explicit gaps** — limitations or future work the authors themselves name\n"
                    f"2. **Implicit gaps** — problems the field has not addressed, visible from "
                    f"   comparing the papers' scope, assumptions, or evaluation protocols\n"
                    f"3. **Methodological gaps** — things these works assume away or leave untested\n\n"
                    f"For each gap, state: what is missing, why it matters, and what a concrete "
                    f"research question to address it might look like. "
                    f"Be specific — cite paper numbers where relevant. "
                    f"Output as markdown with clear numbered sections."
                ),
            }
        ]
        return self.chat(
            messages,
            system=(
                "You are a senior research scientist performing a critical literature review. "
                "Your job is to identify what is NOT known, NOT solved, and NOT addressed — "
                "not to describe what existing work does."
            ),
        )

    def generate_research_question(
        self, background: str, papers: list[Paper]
    ) -> str:
        """Generate a research question based on background and papers.

        Args:
            background: Synthesized background context.
            papers: List of relevant papers.

        Returns:
            Generated research question string.
        """
        paper_titles = "\n".join(
            f"- {p.title} ({p.published})" for p in papers[:10]
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Based on the following background and papers, propose 3 compelling "
                    f"research questions that could advance this field.\n\n"
                    f"BACKGROUND:\n{background}\n\n"
                    f"RELEVANT PAPERS:\n{paper_titles}\n\n"
                    f"For each question, explain why it is important and how it could be "
                    f"investigated. Format each question clearly numbered."
                ),
            }
        ]
        return self.chat(
            messages,
            system=(
                "You are an expert scientific researcher. Generate insightful, specific, "
                "and feasible research questions that push the boundaries of current knowledge."
            ),
        )

    def analyze_papers(self, papers: list[Paper], focus: str) -> str:
        """Analyze a set of papers with a given focus.

        Args:
            papers: List of Paper objects to analyze.
            focus: Specific focus or question for the analysis.

        Returns:
            Analysis text string.
        """
        paper_summaries = "\n\n".join(
            f"Title: {p.title}\nAuthors: {p.short_authors()}\nPublished: {p.published}\n"
            f"Abstract: {p.abstract[:500]}..."
            for p in papers[:8]
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Analyze the following papers with focus on: {focus}\n\n"
                    f"PAPERS:\n{paper_summaries}\n\n"
                    f"Provide:\n"
                    f"1. Key themes and trends\n"
                    f"2. Methodological approaches used\n"
                    f"3. Gaps and open problems\n"
                    f"4. Most significant findings\n"
                    f"5. How these papers relate to the focus area"
                ),
            }
        ]
        return self.chat(
            messages,
            system=(
                "You are an expert academic researcher with deep knowledge across scientific domains. "
                "Provide thorough, critical analysis of research papers."
            ),
        )

    def plan_experiments(
        self, question: str, context: str
    ) -> list[dict]:
        """Generate an experiment plan for a research question.

        Args:
            question: The research question to investigate.
            context: Background context from literature review.

        Returns:
            List of experiment dicts with keys: title, description, code_template.
        """
        import json as _json

        messages = [
            {
                "role": "user",
                "content": (
                    f"Research Question: {question}\n\n"
                    f"Background Context:\n{context[:1500]}\n\n"
                    f"Generate 3-5 concrete experiments to investigate this research question. "
                    f"Each experiment should be feasible with Python code and publicly available data.\n\n"
                    f"Respond with a JSON array where each item has:\n"
                    f"- title: short experiment name\n"
                    f"- description: what the experiment does and what it tests\n"
                    f"- code_template: complete, runnable Python code (use matplotlib, numpy, "
                    f"sklearn, pandas, or similar standard libraries)\n\n"
                    f"The code should be self-contained and print results to stdout."
                ),
            }
        ]
        response = self.chat(
            messages,
            system=(
                "You are an expert computational scientist. Design rigorous experiments "
                "that directly address research questions. Always provide complete, "
                "runnable Python code. Respond ONLY with valid JSON."
            ),
        )

        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            experiments = _json.loads(text)
            if isinstance(experiments, list):
                return experiments
        except _json.JSONDecodeError:
            pass

        # Fallback: return a single placeholder experiment
        return [
            {
                "title": "Exploratory Analysis",
                "description": f"Investigate: {question}",
                "code_template": (
                    "import numpy as np\nimport matplotlib\nmatplotlib.use('Agg')\n"
                    "import matplotlib.pyplot as plt\n\n"
                    "# Placeholder experiment code\n"
                    "print('Running exploratory analysis...')\n"
                    "x = np.linspace(0, 10, 100)\n"
                    "y = np.sin(x)\n"
                    "print(f'Analysis complete. Data range: [{x.min():.2f}, {x.max():.2f}]')\n"
                ),
            }
        ]

    def generate_conclusion(self, investigation: Investigation) -> str:
        """Generate a structured research summary for an investigation.

        Args:
            investigation: The completed Investigation object.

        Returns:
            Formatted research paper text (Abstract + sections).
        """
        paper_refs = "\n".join(
            f"[{i+1}] {p.title} - {p.short_authors(2)} ({p.published})"
            for i, p in enumerate(investigation.papers[:10])
        )

        exp_results = "\n\n".join(
            f"Experiment: {e.title}\nStatus: {e.status}\nResult: {e.result[:300] if e.result else 'No results recorded.'}"
            for e in investigation.experiments
        ) or "No experiments were run."

        messages = [
            {
                "role": "user",
                "content": (
                    f"Write a structured research summary for the following investigation:\n\n"
                    f"TITLE: {investigation.title}\n"
                    f"RESEARCH QUESTION: {investigation.research_question or 'Not defined'}\n\n"
                    f"BACKGROUND:\n{investigation.user_context[:800] if investigation.user_context else 'No background recorded.'}\n\n"
                    f"EXPERIMENTS AND RESULTS:\n{exp_results}\n\n"
                    f"CITED PAPERS:\n{paper_refs}\n\n"
                    f"Write a complete research summary with these sections:\n"
                    f"# Abstract\n# Introduction\n# Background\n# Methods\n# Results\n"
                    f"# Discussion\n# Conclusion\n# References\n\n"
                    f"Use markdown formatting. Be specific and scientific. "
                    f"Base conclusions on the actual experimental results provided."
                ),
            }
        ]
        return self.chat(
            messages,
            system=(
                "You are an expert scientific writer. Write clear, rigorous research summaries "
                "in the style of top academic journals. Be specific, evidence-based, and "
                "acknowledge limitations honestly."
            ),
        )

    def synthesize_papers(self, papers: list[Paper], topic: str) -> str:
        """Synthesize background context from a list of papers.

        Args:
            papers: Papers to synthesize.
            topic: The research topic for context.

        Returns:
            Synthesized background text.
        """
        summaries = "\n\n".join(
            f"Title: {p.title}\nPublished: {p.published}\nAbstract: {p.abstract[:400]}"
            for p in papers[:10]
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Synthesize the following papers into a coherent background section "
                    f"for research on: {topic}\n\n"
                    f"PAPERS:\n{summaries}\n\n"
                    f"Write a flowing synthesis that:\n"
                    f"1. Identifies the main research threads\n"
                    f"2. Shows how the works relate to each other\n"
                    f"3. Identifies the current state of the art\n"
                    f"4. Points to open problems\n"
                    f"5. Sets up the motivation for new research\n\n"
                    f"Keep it focused and scientifically precise (400-600 words)."
                ),
            }
        ]
        return self.chat(
            messages,
            system=(
                "You are an expert scientific researcher writing a literature review. "
                "Synthesize papers concisely and identify key insights and gaps."
            ),
        )

    def interpret_experiment_result(
        self, experiment_title: str, code: str, result: str, research_question: str
    ) -> str:
        """Interpret an experiment result and suggest next steps.

        Args:
            experiment_title: Title of the experiment.
            code: The code that was run.
            result: Output of the experiment.
            research_question: The overarching research question.

        Returns:
            Interpretation and next-steps text.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Research Question: {research_question}\n\n"
                    f"Experiment: {experiment_title}\n\n"
                    f"Code Run:\n```python\n{code[:800]}\n```\n\n"
                    f"Output:\n{result[:1000]}\n\n"
                    f"Interpret these results:\n"
                    f"1. What do the results tell us?\n"
                    f"2. Do they support or challenge hypotheses?\n"
                    f"3. What are the limitations?\n"
                    f"4. What should be investigated next?"
                ),
            }
        ]
        return self.chat(
            messages,
            system=(
                "You are an expert scientist analyzing experimental results. "
                "Provide clear, evidence-based interpretation."
            ),
        )
