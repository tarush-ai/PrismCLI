"""Abstract base for Thinker/Judge model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from prism.models import NoveltyScore, Paper, ProposedIdea


class ModelBackend(ABC):
    """Interface for the Thinker and Judge model backends.

    The CLI never loads model weights directly. It calls this interface,
    and the concrete backend decides whether to hit a local vllm server,
    a remote endpoint, or a mock.
    """

    @abstractmethod
    def explore(
        self,
        paper: Paper,
        num_ideas: int = 3,
        use_future_explorations: bool = True,
    ) -> list[ProposedIdea]:
        """Run Thinker's explore on a paper.

        Args:
            paper: The reference paper to explore.
            num_ideas: Number of follow-up ideas to generate.
            use_future_explorations: If True and paper.future_explorations is
                populated, use the richer prompt that includes future work context.

        Returns:
            List of ProposedIdea objects parsed from the model's structured output.
        """
        ...

    @abstractmethod
    def judge(
        self,
        idea_title: str,
        idea_abstract: str,
        reference_paper: Paper,
    ) -> NoveltyScore:
        """Run Judge to compare a proposed idea against a reference paper.

        Args:
            idea_title: Title of the proposed research idea (treated as Paper A).
            idea_abstract: Abstract of the proposed idea.
            reference_paper: The reference paper to compare against (Paper B).

        Returns:
            NoveltyScore with winner, reasoning, and score.
        """
        ...

    def is_available(self) -> bool:
        """Return True if the backend is reachable."""
        return True
