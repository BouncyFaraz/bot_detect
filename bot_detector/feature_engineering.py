"""Feature engineering for the bot detector."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Callable, Dict, List, Sequence

from .data_collection import SocialEvent


FeatureRow = Dict[str, float]
TextFeatureBuilder = Callable[[Sequence[SocialEvent]], FeatureRow]
BehaviourFeatureBuilder = Callable[[Sequence[SocialEvent]], FeatureRow]


@dataclass
class UserFeatureVector:
    """Bundle a user identifier with its engineered features."""

    user_id: str
    features: FeatureRow

SPAM_KEYWORDS = ("buy followers", "crypto", "hot deals", "🔥")


def average_post_length(events: Sequence[SocialEvent]) -> FeatureRow:
    lengths = [len(event.text) for event in events]
    return {"avg_post_length": float(mean(lengths)) if lengths else 0.0}


def emoji_ratio(events: Sequence[SocialEvent]) -> FeatureRow:
    emoji_count = sum(sum(1 for char in event.text if char >= "\U0001F300") for event in events)
    total_chars = sum(len(event.text) for event in events)
    ratio = (emoji_count / total_chars) if total_chars else 0.0
    return {"emoji_ratio": ratio}


def keyword_hits(events: Sequence[SocialEvent]) -> FeatureRow:
    hits = 0
    for event in events:
        text_lower = event.text.lower()
        hits += sum(1 for keyword in SPAM_KEYWORDS if keyword in text_lower)
    return {"keyword_hits": float(hits)}


def posting_interval_stats(events: Sequence[SocialEvent]) -> FeatureRow:
    timestamps = sorted(event.timestamp for event in events)
    deltas: List[float] = []
    for first, second in zip(timestamps, timestamps[1:]):
        deltas.append((second - first).total_seconds() / 60.0)
    if not deltas:
        return {"avg_post_interval": 0.0, "post_interval_stdev": 0.0}
    return {
        "avg_post_interval": float(mean(deltas)),
        "post_interval_stdev": float(pstdev(deltas)),
    }


def account_age_days(events: Sequence[SocialEvent]) -> FeatureRow:
    now = max((event.timestamp for event in events), default=datetime.now(timezone.utc))
    ages = [
        (now - event.account_created_at).total_seconds() / (60 * 60 * 24)
        for event in events
    ]
    return {"avg_account_age_days": float(mean(ages)) if ages else 0.0}


def follower_following_ratio(events: Sequence[SocialEvent]) -> FeatureRow:
    latest = sorted(events, key=lambda event: event.timestamp)[-1]
    following = max(1, latest.following)
    ratio = latest.followers / following
    return {
        "follower_following_ratio": float(ratio),
        "follower_count": float(latest.followers),
        "following_count": float(latest.following),
    }


@dataclass
class FeatureEngineer:
    """Combine individual feature builders to produce feature tables."""

    text_feature_builders: Sequence[TextFeatureBuilder]
    behaviour_feature_builders: Sequence[BehaviourFeatureBuilder]

    def build_feature_rows(self, events: Sequence[SocialEvent]) -> List[FeatureRow]:
        grouped = self._group_by_user(events)
        return [self._build_feature_row(user_events) for user_events in grouped.values()]

    def build_feature_table(self, events: Sequence[SocialEvent]) -> List[UserFeatureVector]:
        grouped = self._group_by_user(events)
        table = []
        for user_id in sorted(grouped):
            feature_row = self._build_feature_row(grouped[user_id])
            table.append(UserFeatureVector(user_id=user_id, features=feature_row))
        return table

    def build_user_feature(self, events: Sequence[SocialEvent]) -> FeatureRow:
        return self._build_feature_row(events)

    def _build_feature_row(self, events: Sequence[SocialEvent]) -> FeatureRow:
        feature_row: FeatureRow = {}
        for builder in self.text_feature_builders:
            feature_row.update(builder(events))
        for builder in self.behaviour_feature_builders:
            feature_row.update(builder(events))
        return feature_row

    @staticmethod
    def _group_by_user(events: Sequence[SocialEvent]) -> Dict[str, List[SocialEvent]]:
        grouped: Dict[str, List[SocialEvent]] = defaultdict(list)
        for event in events:
            grouped[event.user_id].append(event)
        return grouped
