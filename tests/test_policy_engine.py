from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AppSettings  # noqa: E402
from app.schemas import FinalStatus, WindowStatus  # noqa: E402
from app.services.policy import decide_final_status  # noqa: E402


def _settings(policy_mode: str, manual_affects: bool = True) -> AppSettings:
    return AppSettings(
        demo_mode=True,
        windows=[],
        policy_mode=policy_mode,
        k_of_n=(3, 8),
        match_threshold=0.42,
        manual_affects_final=manual_affects,
        end_class_face_fraction=0.1,
        end_class_empty_seconds=120,
    )


def test_mid_dominant_mid_wins():
    s = _settings("mid_dominant")
    result = decide_final_status(
        s,
        early=WindowStatus.NOT_SEEN,
        mid=WindowStatus.MATCHED,
        manual=None,
    )
    assert result == FinalStatus.PRESENT


def test_mid_dominant_conflict_review():
    s = _settings("mid_dominant")
    result = decide_final_status(
        s,
        early=WindowStatus.MATCHED,
        mid=WindowStatus.NOT_SEEN,
        manual=None,
    )
    assert result == FinalStatus.REVIEW


def test_both_required_needs_both_present():
    s = _settings("both_required")
    assert decide_final_status(
        s,
        early=WindowStatus.MATCHED,
        mid=WindowStatus.MATCHED,
        manual=None,
    ) == FinalStatus.PRESENT
    assert decide_final_status(
        s,
        early=WindowStatus.MATCHED,
        mid=WindowStatus.NOT_SEEN,
        manual=None,
    ) == FinalStatus.REVIEW


def test_either_window_simple():
    s = _settings("either_window")
    assert decide_final_status(
        s,
        early=WindowStatus.MATCHED,
        mid=WindowStatus.NOT_SEEN,
        manual=None,
    ) == FinalStatus.PRESENT
    assert decide_final_status(
        s,
        early=WindowStatus.NOT_SEEN,
        mid=WindowStatus.NOT_SEEN,
        manual=None,
    ) == FinalStatus.ABSENT


def test_manual_override_present_and_absent():
    s = _settings("either_window", manual_affects=True)
    assert decide_final_status(
        s,
        early=WindowStatus.NOT_SEEN,
        mid=WindowStatus.NOT_SEEN,
        manual=WindowStatus.MATCHED,
    ) == FinalStatus.PRESENT
    assert decide_final_status(
        s,
        early=WindowStatus.MATCHED,
        mid=WindowStatus.MATCHED,
        manual=WindowStatus.NOT_SEEN,
    ) == FinalStatus.ABSENT
