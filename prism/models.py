"""Pydantic models for Prism data structures."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Represents an academic paper."""

    id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: str          # ISO date string "YYYY-MM-DD"
    arxiv_id: str = ""
    future_explorations: str = ""   # "Future Work" section if extractable
    citation_count: int = 0

    def short_authors(self, max_authors: int = 3) -> str:
        """Return a shortened author list."""
        if len(self.authors) <= max_authors:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:max_authors]) + f" et al. ({len(self.authors)} total)"

    def year(self) -> int:
        """Return publication year as int."""
        try:
            return int(self.published[:4])
        except (ValueError, IndexError):
            return 0


class ProposedIdea(BaseModel):
    """A research idea proposed by the Thinker model."""

    id: str
    title: str
    abstract: str
    source_paper_id: str        # which reference paper spawned this
    thinking_trace: str = ""    # Thinker's chain-of-thought; stored, hidden by default
    selected: bool = False
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class NoveltyScore(BaseModel):
    """Judge's comparison of a proposed idea against a reference paper."""

    idea_title: str
    reference_paper_id: str
    reference_paper_title: str
    winner: Literal["idea", "paper"]
    judge_reasoning: str
    score: float = 0.0          # running win-rate; 0.0–1.0, higher = more novel


class Experiment(BaseModel):
    """Represents a research experiment."""

    id: str
    investigation_id: str
    title: str
    description: str
    hypothesis: str = ""
    code: str = ""
    status: Literal["pending", "running", "complete", "failed", "skipped"] = "pending"
    result: str = ""
    interpretation: str = ""
    is_isolated: bool = True    # if False, a surprise result may pivot remaining plan
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def mark_running(self) -> None:
        self.status = "running"
        self.updated_at = datetime.utcnow().isoformat()

    def mark_complete(self, result: str) -> None:
        self.status = "complete"
        self.result = result
        self.updated_at = datetime.utcnow().isoformat()

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.result = error
        self.updated_at = datetime.utcnow().isoformat()

    def mark_skipped(self) -> None:
        self.status = "skipped"
        self.updated_at = datetime.utcnow().isoformat()


class Investigation(BaseModel):
    """Represents a research investigation."""

    id: str
    title: str
    topic: str
    phase: Literal[
        "observation", "research", "question", "experiment", "conclusion"
    ] = "observation"
    visibility: Literal["private", "public"] = "private"
    user_context: str = ""
    papers: list[Paper] = Field(default_factory=list)
    thinker_ideas: list[ProposedIdea] = Field(default_factory=list)
    research_question: Optional[str] = None
    experiments: list[Experiment] = Field(default_factory=list)
    novelty_scores: list[NoveltyScore] = Field(default_factory=list)
    conclusion: str = ""
    bibtex: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def advance_phase(self) -> None:
        phases = ["observation", "research", "question", "experiment", "conclusion"]
        current_idx = phases.index(self.phase)
        if current_idx < len(phases) - 1:
            self.phase = phases[current_idx + 1]  # type: ignore[assignment]
        self.updated_at = datetime.utcnow().isoformat()

    def touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()

    def next_phase(self) -> str | None:
        phases = ["observation", "research", "question", "experiment", "conclusion"]
        current_idx = phases.index(self.phase)
        if current_idx < len(phases) - 1:
            return phases[current_idx + 1]
        return None

    def phase_index(self) -> int:
        phases = ["observation", "research", "question", "experiment", "conclusion"]
        return phases.index(self.phase)

    def selected_idea(self) -> ProposedIdea | None:
        """Return the user-selected ProposedIdea, or None."""
        for idea in self.thinker_ideas:
            if idea.selected:
                return idea
        return None

    def novelty_win_rate(self) -> float | None:
        """Return average win-rate across all Judge comparisons, or None if no scores."""
        if not self.novelty_scores:
            return None
        wins = sum(1 for s in self.novelty_scores if s.winner == "idea")
        return wins / len(self.novelty_scores)
