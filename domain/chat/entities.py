from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatResponse:
    """Structured response from the chat engine."""

    answer: str
    sources: list[str] = field(default_factory=list)
