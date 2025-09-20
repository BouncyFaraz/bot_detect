"""Pipeline orchestration for the bot detector system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

from . import (
    data_collection,
    feature_engineering,
    inference,
    modeling,
    monitoring,
    preprocessing,
)


@dataclass
class BotDetectionPipeline:
    """Coordinates the major subsystems of the bot detector."""

    collector: data_collection.DataCollector
    preprocessor: preprocessing.Preprocessor
    feature_engineer: feature_engineering.FeatureEngineer
    model: modeling.Model
    inference_service: inference.InferenceService
    monitor: monitoring.Monitor

    @classmethod
    def default(cls) -> "BotDetectionPipeline":
        """Create a pipeline with opinionated defaults suitable for demos."""
        collector = data_collection.MockSocialStream()
        preprocessor = preprocessing.Preprocessor(
            text_normalizers=[
                preprocessing.strip_urls,
                preprocessing.to_lower,
                preprocessing.collapse_whitespace,
            ],
            account_normalizers=[
                preprocessing.normalize_timezone,
            ],
        )
        feature_engineer = feature_engineering.FeatureEngineer(
            text_feature_builders=[
                feature_engineering.average_post_length,
                feature_engineering.emoji_ratio,
                feature_engineering.keyword_hits,
            ],
            behaviour_feature_builders=[
                feature_engineering.posting_interval_stats,
                feature_engineering.account_age_days,
                feature_engineering.follower_following_ratio,
            ],
        )
        model = modeling.RuleBasedModel()
        inference_service = inference.InferenceService(model=model)
        monitor = monitoring.Monitor()
        return cls(
            collector=collector,
            preprocessor=preprocessor,
            feature_engineer=feature_engineer,
            model=model,
            inference_service=inference_service,
            monitor=monitor,
        )

    def run(self, events: Sequence[data_collection.SocialEvent] | None = None) -> inference.InferenceResult:
        """Execute the end-to-end inference loop and update monitoring."""
        if events is None:
            events = list(self.collector.collect())
        preprocessed = self.preprocessor.process_batch(events)
        feature_vectors = self.feature_engineer.build_feature_table(preprocessed)
        inference_result = self.inference_service.score(feature_vectors)
        self.monitor.update(events, inference_result)
        return inference_result

    def update_model(self, labeled_events: Iterable[data_collection.LabeledEvent]) -> None:
        """Trigger model retraining given newly labeled data."""
        grouped_events: Dict[str, list[data_collection.SocialEvent]] = {}
        labels: Dict[str, int] = {}
        for labeled in labeled_events:
            processed = self.preprocessor.process_event(labeled.event)
            grouped_events.setdefault(processed.user_id, []).append(processed)
            labels[processed.user_id] = labeled.label
        if not grouped_events:
            return
        feature_rows = [
            self.feature_engineer.build_user_feature(events)
            for events in grouped_events.values()
        ]
        label_values = [labels[user_id] for user_id in grouped_events.keys()]
        self.model.fit(feature_rows, label_values)
        self.monitor.log_retraining(len(label_values))
