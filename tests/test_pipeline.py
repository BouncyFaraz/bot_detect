from datetime import datetime, timezone

from bot_detector.data_collection import LabeledEvent, MockSocialStream
from bot_detector.pipeline import BotDetectionPipeline


def test_pipeline_produces_probabilities():
    pipeline = BotDetectionPipeline.default()
    result = pipeline.run()
    assert len(result.probabilities) > 0
    assert len(result.user_ids) == len(result.probabilities)
    assert all(0.0 <= prob <= 1.0 for prob in result.probabilities)
    # Ensure monitoring collected stats
    assert pipeline.monitor.average_score() > 0.0


def test_model_update_changes_thresholds():
    pipeline = BotDetectionPipeline.default()
    stream = MockSocialStream(now=datetime(2024, 1, 1, tzinfo=timezone.utc))
    events = list(stream.collect())
    labeled = [
        LabeledEvent(event=events[0], label=0),  # user-0
        LabeledEvent(event=events[2], label=1),  # user-1
        LabeledEvent(event=events[4], label=0),  # user-2
    ]
    previous_max_interval = pipeline.model.max_post_interval_minutes
    pipeline.update_model(labeled)
    assert pipeline.model.max_post_interval_minutes >= previous_max_interval
    assert pipeline.monitor.alerts[-1].startswith("Model retrained")
