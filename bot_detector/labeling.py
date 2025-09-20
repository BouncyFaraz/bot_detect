"""Helpers for managing labeled data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .data_collection import LabeledEvent


@dataclass
class LabelRepository:
    """In-memory storage for labeled events."""

    labels: Dict[str, LabeledEvent] = field(default_factory=dict)

    def upsert(self, labeled_events: Iterable[LabeledEvent]) -> None:
        for labeled_event in labeled_events:
            self.labels[labeled_event.event.user_id] = labeled_event

    def count(self) -> int:
        return len(self.labels)

    def export(self) -> List[LabeledEvent]:
        return list(self.labels.values())
