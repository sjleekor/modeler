"""F1b constants. This module imports no data readers and performs no I/O."""

from datetime import datetime
from zoneinfo import ZoneInfo

from modeler.feedback.core import FeedbackError, require_timestamp

RULE_ID = "kr-v1-post-publication-h20-v1"
RULE_SHA256 = "a475c1f72ec7bb8c1073991c5e768285452ac59d4df91fabd9199fbbb3c4e3a1"
MODEL_ID = "kr_daily_h20_v1"
MODEL_VERSION = "1.0.0"
BUNDLE_SHA256 = "8a97ee5700a4ad8434ab9f85c3e0a213d996b3135140a36e7c921cb26f255375"
REPORT_START = "2026-11-04"
REPORT_END = "2027-01-29"
SEOUL = ZoneInfo("Asia/Seoul")
FIRST_REVIEW = datetime(2026, 12, 4, 13, tzinfo=SEOUL)
FINAL_REVIEW = datetime(2027, 3, 5, 13, tzinfo=SEOUL)
HORIZON = 20
ROUND_TRIP_COST = 0.006
INDEX_KEYS = {"KOSPI": ("kospi", "코스피"), "KOSDAQ": ("kosdaq", "코스닥")}


class OutcomeError(FeedbackError):
    """The frozen outcome/input contract was not met."""


def stamp(value):
    return datetime.fromisoformat(require_timestamp(value).replace("Z", "+00:00"))


def require_real_review(now):
    """Must run before opening a request, index, extract, or price file."""
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise OutcomeError("review clock requires a timezone")
    local = now.astimezone(SEOUL)
    if local < FIRST_REVIEW or local.weekday() != 4 or local.hour < 13:
        raise OutcomeError("real outcome access is outside the registered Friday review window")
    return local


def rule_pin():
    return {
        "rule_id": RULE_ID,
        "rule_sha256": RULE_SHA256,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "bundle_sha256": BUNDLE_SHA256,
        "report_start": REPORT_START,
        "report_end": REPORT_END,
        "horizon_sessions": HORIZON,
        "round_trip_cost": ROUND_TRIP_COST,
    }
