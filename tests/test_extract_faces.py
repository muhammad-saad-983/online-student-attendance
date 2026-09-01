from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.config import EnrollmentSettings
from vision.datasets import extract_faces_for_all, extract_faces_for_person, get_face_detector
from app.services.roster_service import load_roster


def make_settings(tmp_path: Path) -> EnrollmentSettings:
    return EnrollmentSettings(
        data_root=tmp_path / "raw",
        faces_root=tmp_path / "faces",
        frame_stride=1,
        min_face_size=60,
        face_size=64,
        max_frames_per_person=5,
        max_faces_per_frame=1,
        face_margin=0.1,
    )


def test_extract_faces_for_all_handles_empty(tmp_path: Path):
    settings = make_settings(tmp_path)
    # ensure roots exist
    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.faces_root.mkdir(parents=True, exist_ok=True)

    # Should not raise even if no videos exist for roster people
    extract_faces_for_all(settings)

    # Faces directory paths should be created for roster entries (if function touched them)
    roster = load_roster(Path("data/enrollment/roster.csv"))
    expected_person_dirs = [settings.faces_root / student.person_id for student in roster]
    for path in expected_person_dirs:
        assert path.exists()


def test_extract_faces_for_person_no_videos(tmp_path: Path):
    settings = make_settings(tmp_path)
    person_id = "personX"
    person_dir = settings.data_root / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    ensure_faces_dir = settings.faces_root / person_id
    ensure_faces_dir.mkdir(parents=True, exist_ok=True)

    detector = get_face_detector()
    count = extract_faces_for_person(person_id, settings=settings, detector=detector)
    assert count == 0
    # No files should be created
    assert len(list(ensure_faces_dir.glob("*.jpg"))) == 0
