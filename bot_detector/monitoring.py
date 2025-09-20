"""Monitoring utilities for the bot detector."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List

from .data_collection import SocialEvent
from .inference import InferenceResult


@dataclass
class Monitor:
    """Tracks inference statistics for observability."""

    window_size: int = 100
    last_scores: Deque[float] = field(default_factory=deque)
    alerts: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.last_scores = deque(maxlen=self.window_size)

    def update(self, events: Iterable[SocialEvent], result: InferenceResult) -> None:
        for probability in result.probabilities:
            self.last_scores.append(probability)
        if self.last_scores and sum(prob >= 0.8 for prob in self.last_scores) / len(self.last_scores) > 0.4:
            self.alerts.append("High proportion of suspected bots detected.")

    def log_retraining(self, labeled_count: int) -> None:
        self.alerts.append(f"Model retrained with {labeled_count} labeled examples.")

    def average_score(self) -> float:
        if not self.last_scores:
            return 0.0
        return sum(self.last_scores) / len(self.last_scores)
