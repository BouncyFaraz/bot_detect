"""Data collection utilities for social media streams."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Protocol


@dataclass
class SocialEvent:
    """A single social media activity event."""

    user_id: str
    text: str
    timestamp: datetime
    timezone_offset_minutes: int
    followers: int
    following: int
    account_created_at: datetime


@dataclass
class LabeledEvent:
    """Wrap a social event together with a human-provided label."""

    event: SocialEvent
    label: int  # 1 for bot, 0 for human


class DataCollector(Protocol):
    """Interface describing collector behaviour."""

    def collect(self) -> Iterable[SocialEvent]:
        """Return an iterable of social events."""
        raise NotImplementedError


class MockSocialStream:
    """Deterministic stream that yields example social events for demos/tests."""

    def __init__(self, now: datetime | None = None) -> None:
        self._now = now or datetime.now(timezone.utc)

    def collect(self) -> Iterator[SocialEvent]:
        base = self._now
        baseline_account_creation = base - timedelta(days=365)
        for user_idx in range(3):
            for post_idx in range(2):
                yield SocialEvent(
                    user_id=f"user-{user_idx}",
                    text=self._make_text(user_idx, post_idx),
                    timestamp=base - timedelta(minutes=user_idx * 90 + post_idx * 5),
                    timezone_offset_minutes=0 if user_idx % 2 == 0 else -300,
                    followers=20 + user_idx * 7,
                    following=10 + user_idx * 4,
                    account_created_at=baseline_account_creation - timedelta(days=user_idx * 40),
                )

    @staticmethod
    def _make_text(user_idx: int, post_idx: int) -> str:
        if user_idx == 1:
            return "Buy followers now!!! http://spam.example"
        if user_idx == 2 and post_idx == 0:
            return "🔥🔥🔥 Hot deals on crypto!!!"
        return "Just enjoying another day on the platform."
