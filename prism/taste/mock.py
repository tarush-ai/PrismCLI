"""Mock backend — returns fixture data. Used when no backend is configured."""

from __future__ import annotations

import uuid

from prism.taste.base import ModelBackend
from prism.models import NoveltyScore, Paper, ProposedIdea


class MockBackend(ModelBackend):
    """Returns plausible-looking placeholder data without calling any model.

    Used when:
      - prism serve is not running and backend.type = local
      - backend.type = mock (explicit testing mode)
    """

    def is_available(self) -> bool:
        return True

    def explore(
        self,
        paper: Paper,
        num_ideas: int = 3,
        use_future_explorations: bool = True,
    ) -> list[ProposedIdea]:
        ideas = []
        templates = [
            (
                f"Extending {paper.title[:40]}... to Broader Domains",
                f"This work extends the core methodology of the referenced paper to broader "
                f"application domains. We identify key limitations in the original scope and "
                f"propose a generalised framework that demonstrates improved transferability. "
                f"[Backend not configured — start prism serve or set backend.type in config.]",
            ),
            (
                f"Efficiency Improvements in {paper.title[:35]}...",
                f"We address the computational overhead inherent to the original approach by "
                f"introducing a lightweight variant that achieves comparable performance at "
                f"significantly reduced cost. [Mock output — configure a Thinker backend.]",
            ),
            (
                f"Theoretical Analysis of {paper.title[:38]}...",
                f"We provide a formal theoretical characterisation of the proposed method, "
                f"establishing convergence guarantees and sample complexity bounds absent from "
                f"the original work. [Mock output — configure a Thinker backend.]",
            ),
        ]
        for i, (title, abstract) in enumerate(templates[:num_ideas]):
            ideas.append(ProposedIdea(
                id=str(uuid.uuid4()),
                title=title,
                abstract=abstract,
                source_paper_id=paper.id,
                thinking_trace="[Mock backend — no real thinking trace]",
            ))
        return ideas

    def judge(
        self,
        idea_title: str,
        idea_abstract: str,
        reference_paper: Paper,
    ) -> NoveltyScore:
        return NoveltyScore(
            idea_title=idea_title,
            reference_paper_id=reference_paper.id,
            reference_paper_title=reference_paper.title,
            winner="idea",
            judge_reasoning=(
                "[Mock backend — no real Judge scoring. "
                "Configure a Judge backend via prism serve or backend.base_url.]"
            ),
            score=0.5,
        )
