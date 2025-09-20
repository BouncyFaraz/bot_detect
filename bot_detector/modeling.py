"""Lightweight modeling abstractions for bot detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence

from .feature_engineering import FeatureRow


class Model(Protocol):
    """Model interface for inference and training."""

    def predict_proba(self, features: Sequence[FeatureRow]) -> List[float]:
        raise NotImplementedError

    def fit(self, features: Sequence[FeatureRow], labels: Sequence[int]) -> None:
        raise NotImplementedError


@dataclass
class RuleBasedModel:
    """Simple heuristic scoring model to bootstrap the system."""

    spam_keywords: Sequence[str] = (
        "buy followers",
        "crypto",
        "hot deals",
    )
    max_post_interval_minutes: float = 60.0
    min_account_age_days: float = 60.0
    max_follow_ratio: float = 8.0
    min_followers: float = 15.0

    def predict_proba(self, features: Sequence[FeatureRow]) -> List[float]:
        scores: List[float] = []
        for row in features:
            score = 0.0
            score += self._keyword_score(row)
            score += self._posting_cadence_score(row)
            score += self._account_age_score(row)
            score += self._network_score(row)
            scores.append(max(0.0, min(score, 1.0)))
        return scores

    def fit(self, features: Sequence[FeatureRow], labels: Sequence[int]) -> None:
        """Tune heuristics using majority statistics from provided labels."""
        if not features:
            return
        avg_post_interval = sum(row.get("avg_post_interval", 0.0) for row in features) / len(features)
        avg_account_age = sum(row.get("avg_account_age_days", 0.0) for row in features) / len(features)
        avg_follow_ratio = sum(row.get("follower_following_ratio", 1.0) for row in features) / len(features)
        # Adjust heuristics toward human behaviour (label 0)
        human_count = sum(1 for label in labels if label == 0)
        if human_count:
            human_interval = sum(
                row.get("avg_post_interval", 0.0)
                for row, label in zip(features, labels)
                if label == 0
            ) / human_count
            human_age = sum(
                row.get("avg_account_age_days", 0.0)
                for row, label in zip(features, labels)
                if label == 0
            ) / human_count
            human_follow_ratio = sum(
                row.get("follower_following_ratio", 1.0)
                for row, label in zip(features, labels)
                if label == 0
            ) / human_count
            self.max_post_interval_minutes = max(self.max_post_interval_minutes, human_interval)
            self.min_account_age_days = min(self.min_account_age_days, human_age)
            self.max_follow_ratio = max(self.max_follow_ratio, human_follow_ratio * 2)
        else:
            self.max_post_interval_minutes = max(self.max_post_interval_minutes, avg_post_interval)
            self.min_account_age_days = min(self.min_account_age_days, avg_account_age)
            self.max_follow_ratio = max(self.max_follow_ratio, avg_follow_ratio)

    def _keyword_score(self, row: FeatureRow) -> float:
        text_length = row.get("avg_post_length", 0.0)
        keyword_hits = row.get("keyword_hits", 0.0)
        if keyword_hits:
            return min(0.5, keyword_hits * 0.2 + (0.2 if text_length < 60 else 0.0))
        return 0.0

    def _posting_cadence_score(self, row: FeatureRow) -> float:
        avg_interval = row.get("avg_post_interval", self.max_post_interval_minutes)
        stdev = row.get("post_interval_stdev", 0.0)
        if avg_interval < self.max_post_interval_minutes / 4 and stdev < avg_interval / 2:
            return 0.3
        if avg_interval < self.max_post_interval_minutes / 2:
            return 0.15
        return 0.0

    def _account_age_score(self, row: FeatureRow) -> float:
        age = row.get("avg_account_age_days", self.min_account_age_days)
        if age < self.min_account_age_days / 2:
            return 0.3
        if age < self.min_account_age_days:
            return 0.1
        return 0.0

    def _network_score(self, row: FeatureRow) -> float:
        ratio = row.get("follower_following_ratio", 1.0)
        followers = row.get("follower_count", 0.0)
        following = row.get("following_count", 1.0)
        if followers < self.min_followers and following > followers * 2:
            return 0.2
        if ratio > self.max_follow_ratio:
            return 0.15
        if ratio < 0.2:
            return 0.1
        return 0.0
