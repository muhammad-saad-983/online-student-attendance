from __future__ import annotations

import cv2
from pathlib import Path
from typing import Iterable, List, Sequence

from app.models import Student
from app.services.roster_service import load_roster
from vision.config import EnrollmentSettings


def get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return detector


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_videos(directory: Path) -> Iterable[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _expand_bbox(x: int, y: int, w: int, h: int, margin: float, frame_shape: Sequence[int]) -> tuple[int, int, int, int]:
    img_h, img_w = frame_shape[:2]
    x0 = max(0, int(x - margin * w))
    y0 = max(0, int(y - margin * h))
    x1 = min(img_w, int(x + w + margin * w))
    y1 = min(img_h, int(y + h + margin * h))
    return x0, y0, x1, y1


def extract_faces_for_person(
    person_id: str,
    settings: EnrollmentSettings,
    detector: cv2.CascadeClassifier | None = None,
) -> int:
    """
    Extract faces for a single person from all videos under data_root/person_id,
    write them into faces_root/person_id as JPEGs, and return the number of faces saved.
    """

    person_raw_dir = settings.data_root / person_id
    person_faces_dir = settings.faces_root / person_id
    ensure_dir(person_faces_dir)

    video_files = list(_iter_videos(person_raw_dir))
    if not video_files:
        return 0

    detector = detector or get_face_detector()
    saved_count = 0

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % settings.frame_stride != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(settings.min_face_size, settings.min_face_size),
            )

            if len(faces) == 0:
                frame_idx += 1
                continue

            faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            for (x, y, w, h) in faces_sorted[: settings.max_faces_per_frame]:
                x0, y0, x1, y1 = _expand_bbox(x, y, w, h, settings.face_margin, frame.shape)
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = frame[y0:y1, x0:x1]
                resized = cv2.resize(crop, (settings.face_size, settings.face_size), interpolation=cv2.INTER_AREA)
                out_path = person_faces_dir / f"face_{saved_count:05d}.jpg"
                cv2.imwrite(str(out_path), resized)
                saved_count += 1
                if saved_count >= settings.max_frames_per_person:
                    break

            frame_idx += 1
            if saved_count >= settings.max_frames_per_person:
                break

        cap.release()

        if saved_count >= settings.max_frames_per_person:
            break

    return saved_count


def extract_faces_for_all(settings: EnrollmentSettings) -> None:
    roster_path = Path("data/enrollment/roster.csv")
    roster: List[Student] = load_roster(roster_path)

    detector = get_face_detector()
    total = 0

    for student in roster:
        count = extract_faces_for_person(student.person_id, settings=settings, detector=detector)
        print(f"[extract] {student.person_id} ({student.name}): {count} faces")
        total += count

    print(f"[extract] Total faces extracted: {total}")
