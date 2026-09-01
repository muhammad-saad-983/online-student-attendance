from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class WindowStatus(str, Enum):
    MATCHED = "matched"
    UNCERTAIN = "uncertain"
    NOT_SEEN = "not_seen"


class FinalDecision(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    REVIEW = "review"


class FinalStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    REVIEW = "review"


class FinalOverrideRequest(BaseModel):
    roll_no: str
    final_status: FinalStatus
    reason: Optional[str] = None


class StudentWindowResult(BaseModel):
    roll_no: str
    status: WindowStatus
    distance: Optional[float] = None  # optional similarity metric


class WindowResultPayload(BaseModel):
    window_name: str  # "early" | "mid" | "manual"
    timestamp: datetime
    results: List[StudentWindowResult]


class FinalizeRequest(BaseModel):
    timestamp: datetime


class FinalizeStudentResult(BaseModel):
    roll_no: str
    decision: FinalDecision


class FinalizeResponse(BaseModel):
    decisions: List[FinalizeStudentResult]
    finalized_at: datetime
    policy_mode: str


class BasicResponse(BaseModel):
    ok: bool
    message: str


class TriggerWindowRequest(BaseModel):
    window_name: str = "manual"
