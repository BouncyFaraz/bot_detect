"""Inference orchestration for bot predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .feature_engineering import FeatureRow, UserFeatureVector
from .modeling import Model


@dataclass
class InferenceResult:
    user_ids: List[str]
    probabilities: List[float]

    def top_suspects(self, threshold: float = 0.6) -> List[Tuple[str, float]]:
        return [
            (user_id, prob)
            for user_id, prob in zip(self.user_ids, self.probabilities)
            if prob >= threshold
        ]


@dataclass
class InferenceService:
    model: Model

    def score(self, feature_vectors: Sequence[UserFeatureVector]) -> InferenceResult:
        features = [vector.features for vector in feature_vectors]
        probabilities = self.model.predict_proba(features)
        user_ids = [vector.user_id for vector in feature_vectors]
        return InferenceResult(user_ids=user_ids, probabilities=probabilities)
