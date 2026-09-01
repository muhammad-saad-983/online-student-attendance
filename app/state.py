from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from app.schemas import FinalDecision, FinalStatus, FinalizeStudentResult, WindowStatus


WindowName = str  # "early" | "mid" | "manual"


@dataclass
class WindowInfo:
    status: str = "idle"  # "idle" | "running" | "done"
    last_updated: Optional[datetime] = None


@dataclass
class AttendanceState:
    # (roll_no, window_name) -> WindowStatus
    per_student_window: Dict[Tuple[str, WindowName], WindowStatus] = field(default_factory=dict)
    # window_name -> WindowInfo
    windows: Dict[WindowName, WindowInfo] = field(
        default_factory=lambda: {
            "early": WindowInfo(),
            "mid": WindowInfo(),
            "manual": WindowInfo(),
        }
    )
    class_ended: bool = False
    finalized: bool = False
    finalized_at: Optional[datetime] = None
    final_decisions: Dict[str, FinalDecision] = field(default_factory=dict)
    final_overrides: Dict[str, FinalStatus] = field(default_factory=dict)
    override_reasons: Dict[str, str] = field(default_factory=dict)
    last_action_label: Optional[str] = None
    last_action_timestamp: Optional[datetime] = None

    def reset(self) -> None:
        self.per_student_window.clear()
        self.windows = {
            "early": WindowInfo(),
            "mid": WindowInfo(),
            "manual": WindowInfo(),
        }
        self.class_ended = False
        self.finalized = False
        self.finalized_at = None
        self.final_decisions.clear()
        self.final_overrides.clear()
        self.override_reasons.clear()
        self.last_action_label = None
        self.last_action_timestamp = None

    def record_last_action(self, label: str, timestamp: Optional[datetime] = None) -> None:
        """
        Record a human-readable description of the last action (window completed
        or manual override) and its timestamp.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self.last_action_label = label
        self.last_action_timestamp = timestamp


_state = AttendanceState()


def get_state() -> AttendanceState:
    return _state
