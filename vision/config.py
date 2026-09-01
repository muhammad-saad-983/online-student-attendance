from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class EnrollmentSettings(BaseModel):
    data_root: Path
    faces_root: Path
    frame_stride: int
    min_face_size: int
    face_size: int
    max_frames_per_person: int
    max_faces_per_frame: int
    face_margin: float


SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "config" / "enrollment.yaml"


def load_enrollment_settings() -> EnrollmentSettings:
    raw = yaml.safe_load(SETTINGS_PATH.read_text())
    return EnrollmentSettings(**raw)
