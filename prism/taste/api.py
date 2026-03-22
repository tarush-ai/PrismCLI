"""OpenAI-compatible HTTP backend for Thinker and Judge models.

Talks to any vllm, HuggingFace Inference Endpoint, or Ollama server
that speaks the OpenAI chat completions API. The CLI uses this backend;
no transformers/from_pretrained code ever runs in the CLI process.

Start a local server with:
    prism serve start
which boots vllm with SciThinker-30B and SciJudge-30B.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

import httpx
from openai import OpenAI

from prism.taste.base import ModelBackend
from prism.models import NoveltyScore, Paper, ProposedIdea


def _parse_proposed_ideas(text: str, source_paper_id: str, thinking_trace: str) -> list[ProposedIdea]:
    """Parse structured Thinker output into ProposedIdea objects.

    Expected format:
        ## Proposed Idea 1:
        Title: <title>
        Abstract: <abstract>

        ## Proposed Idea N:
        ...
    """
    ideas: list[ProposedIdea] = []

    # Split on idea headers
    blocks = re.split(r"##\s*Proposed Idea\s*\d+\s*:", text, flags=re.IGNORECASE)
    for block in blocks[1:]:   # skip preamble before first header
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        title = ""
        abstract_lines: list[str] = []
        in_abstract = False

        for line in lines:
            if re.match(r"^Title\s*:", line, re.IGNORECASE):
                title = re.sub(r"^Title\s*:\s*", "", line, flags=re.IGNORECASE).strip()
                in_abstract = False
            elif re.match(r"^Abstract\s*:", line, re.IGNORECASE):
                first_line = re.sub(r"^Abstract\s*:\s*", "", line, flags=re.IGNORECASE).strip()
                if first_line:
                    abstract_lines.append(first_line)
                in_abstract = True
            elif in_abstract:
                # Stop if we hit the next idea block indicator
                if re.match(r"^##", line):
                    break
                abstract_lines.append(line)

        abstract = " ".join(abstract_lines).strip()
        if title and abstract:
            ideas.append(ProposedIdea(
                id=str(uuid.uuid4()),
                title=title,
                abstract=abstract,
                source_paper_id=source_paper_id,
                thinking_trace=thinking_trace,
            ))

    return ideas


def _parse_judge_response(
    response: str,
    idea_title: str,
    reference_paper: Paper,
) -> NoveltyScore:
    """Parse Judge model response into a NoveltyScore.

    The Judge responds with reasoning in <reason>...</reason> and
    a binary answer in <answer>A</answer> or <answer>B</answer>.
    Paper A = the proposed idea, Paper B = the reference paper.
    Winner = "idea" if answer is A, "paper" if B.
    """
    reasoning = ""
    reason_match = re.search(r"<reason>(.*?)</reason>", response, re.DOTALL | re.IGNORECASE)
    if reason_match:
        reasoning = reason_match.group(1).strip()
    else:
        # Fall back to full response as reasoning
        reasoning = response.strip()[:1000]

    answer = "B"  # default: reference paper wins
    answer_match = re.search(r"<answer>\s*([AB])\s*</answer>", response, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip().upper()

    winner: str = "idea" if answer == "A" else "paper"

    return NoveltyScore(
        idea_title=idea_title,
        reference_paper_id=reference_paper.id,
        reference_paper_title=reference_paper.title,
        winner=winner,  # type: ignore[arg-type]
        judge_reasoning=reasoning,
        score=1.0 if winner == "idea" else 0.0,
    )


class APIBackend(ModelBackend):
    """Talks to Thinker and Judge via an OpenAI-compatible HTTP server."""

    def __init__(
        self,
        base_url: str,
        thinker_model: str,
        judge_model: str,
        api_key: str = "EMPTY",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._thinker_model = thinker_model
        self._judge_model = judge_model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=3) as client:
                resp = client.get(f"{self._base_url}/models")
                return resp.status_code == 200
        except Exception:
            return False

    def explore(
        self,
        paper: Paper,
        num_ideas: int = 3,
        use_future_explorations: bool = True,
    ) -> list[ProposedIdea]:
        """Call SciThinker to generate follow-up ideas from a reference paper."""
        has_future = bool(paper.future_explorations.strip())

        if use_future_explorations and has_future:
            user_prompt = (
                f"You are a knowledgeable and insightful AI researcher. You have come across a "
                f"new research paper with the following title and abstract:\n"
                f"Title: {paper.title}\n"
                f"Abstract: {paper.abstract}\n"
                f"Future Explorations: {paper.future_explorations}\n\n"
                f"Based on the core ideas, methods, or findings of this work, engage in heuristic "
                f"thinking and propose {num_ideas} follow-up research ideas. "
                f"You need not confine yourself to the specific scenario or task of the original "
                f"paper. You may consider shortcomings of the original method, propose improvements, "
                f"apply its ideas to other tasks or AI domains, or introduce entirely new problems. "
                f"Aim to formulate ideas with high academic value and potential impact. "
                f"If available, the Future Explorations section may assist in this task.\n\n"
                f"In your response, solely present your proposed title and abstract. "
                f"Ensure no specific numerical results are included in the abstract. "
                f"Ensure the abstract is of moderate length.\n\n"
                f"Output format (strict, no extra text):\n\n"
                f"## Proposed Idea 1:\n"
                f"Title: <your proposed paper title>\n"
                f"Abstract: <your proposed abstract>\n\n"
                f"...\n\n"
                f"## Proposed Idea {num_ideas}:\n"
                f"Title: <your proposed paper title>\n"
                f"Abstract: <your proposed abstract>"
            )
        else:
            user_prompt = (
                f"You are a knowledgeable and insightful AI researcher. You have come across a "
                f"new research paper with the following title and abstract:\n"
                f"Title: {paper.title}\n"
                f"Abstract: {paper.abstract}\n\n"
                f"Based on the core ideas, methods, or findings of this work, engage in heuristic "
                f"thinking and propose {num_ideas} follow-up research ideas. "
                f"You need not confine yourself to the specific scenario or task of the original "
                f"paper. You may consider shortcomings of the original method, propose improvements, "
                f"apply its ideas to other tasks or AI domains, or introduce entirely new problems. "
                f"Aim to formulate ideas with high academic value and potential impact.\n\n"
                f"In your response, solely present your proposed title and abstract. "
                f"Ensure no specific numerical results are included in the abstract. "
                f"Ensure the abstract is of moderate length.\n\n"
                f"Output format (strict, no extra text):\n\n"
                f"## Proposed Idea 1:\n"
                f"Title: <your proposed paper title>\n"
                f"Abstract: <your proposed abstract>\n\n"
                f"...\n\n"
                f"## Proposed Idea {num_ideas}:\n"
                f"Title: <your proposed paper title>\n"
                f"Abstract: <your proposed abstract>"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You first think about the reasoning process "
                    "in your mind and then provide the user with the answer."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._thinker_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=32768,
            temperature=0.6,
            top_p=0.95,
        )

        full_text = response.choices[0].message.content or ""

        # The Thinker model outputs <think>...</think> followed by the structured ideas.
        # Split on the closing think tag (token 151668 in the raw model = </think>).
        thinking_trace = ""
        content = full_text
        think_match = re.search(r"<think>(.*?)</think>(.*)", full_text, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking_trace = think_match.group(1).strip()
            content = think_match.group(2).strip()

        return _parse_proposed_ideas(content, source_paper_id=paper.id, thinking_trace=thinking_trace)

    def judge(
        self,
        idea_title: str,
        idea_abstract: str,
        reference_paper: Paper,
    ) -> NoveltyScore:
        """Call SciJudge to compare a proposed idea (A) against a reference paper (B)."""
        date_string = datetime.now().strftime("%Y-%m-%d")

        user_prompt = (
            f"Today is {date_string}. Based on the titles, abstracts, and publication dates "
            f"of the following two papers A and B, determine which paper has a higher citation count.\n"
            f"Show your reasoning process in <reason> </reason> tags. "
            f"Return the final answer in <answer> </answer> tags. "
            f"The final answer should contain only 'A' or 'B'.\n\n"
            f"## Paper A:\n"
            f"  **Title:** {idea_title}\n"
            f"  **Abstract:** {idea_abstract}\n"
            f"  **Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"## Paper B:\n"
            f"  **Title:** {reference_paper.title}\n"
            f"  **Abstract:** {reference_paper.abstract}\n"
            f"  **Date:** {reference_paper.published}\n"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You first think about the reasoning process "
                    "in your mind and then provide the user with the answer."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._judge_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=2048,
            temperature=0.7,
            top_p=0.8,
        )

        full_text = response.choices[0].message.content or ""
        return _parse_judge_response(full_text, idea_title, reference_paper)
