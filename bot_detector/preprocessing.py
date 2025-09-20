"""Preprocessing helpers for social activity events."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, List, Sequence

from .data_collection import SocialEvent

TextNormalizer = Callable[[str], str]
AccountNormalizer = Callable[[SocialEvent], SocialEvent]

URL_RE = re.compile(r"https?://\\S+")


def strip_urls(text: str) -> str:
    """Remove URLs from text."""
    return URL_RE.sub("", text)


def to_lower(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip()


def normalize_timezone(event: SocialEvent) -> SocialEvent:
    """Convert timezone-aware timestamps to UTC and set offset to zero."""
    tz_offset = event.timezone_offset_minutes
    normalized_timestamp = event.timestamp - timedelta(minutes=tz_offset)
    return SocialEvent(
        user_id=event.user_id,
        text=event.text,
        timestamp=normalized_timestamp,
        timezone_offset_minutes=0,
        followers=event.followers,
        following=event.following,
        account_created_at=event.account_created_at,
    )


@dataclass
class Preprocessor:
    """Apply text and account normalizations to events."""

    text_normalizers: Sequence[TextNormalizer]
    account_normalizers: Sequence[AccountNormalizer]

    def process_event(self, event: SocialEvent) -> SocialEvent:
        """Normalize a single event."""
        text = event.text
        for normalizer in self.text_normalizers:
            text = normalizer(text)
        normalized_event = SocialEvent(
            user_id=event.user_id,
            text=text,
            timestamp=event.timestamp,
            timezone_offset_minutes=event.timezone_offset_minutes,
            followers=event.followers,
            following=event.following,
            account_created_at=event.account_created_at,
        )
        for normalizer in self.account_normalizers:
            normalized_event = normalizer(normalized_event)
        return normalized_event

    def process_batch(self, events: Sequence[SocialEvent]) -> List[SocialEvent]:
        return [self.process_event(event) for event in events]
