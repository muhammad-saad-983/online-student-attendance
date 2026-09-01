from pathlib import Path
import sys
from datetime import datetime

from fastapi.encoders import jsonable_encoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import Student  # noqa: E402
from app.schemas import StudentWindowResult, WindowResultPayload, WindowStatus  # noqa: E402
from vision.windows import FrameVotes  # noqa: E402


def test_frame_votes_register_and_counts():
    votes = FrameVotes()
    votes.register("id1", 0.1)
    votes.register("id1", 0.2)
    votes.register("id2", 0.3)

    assert votes.counts["id1"] == 2
    assert votes.counts["id2"] == 1
    assert abs(votes.best_distance["id1"] - 0.1) < 1e-6


def test_k_of_n_decision_simple():
    k = 2
    counts = {"id1": 3, "id2": 1, "id3": 0}
    best_distance = {"id1": 0.2, "id2": 0.5}

    students = [
        Student(roll_no="r1", person_id="id1", name="S1"),
        Student(roll_no="r2", person_id="id2", name="S2"),
        Student(roll_no="r3", person_id="id3", name="S3"),
    ]

    results = []
    for student in students:
        identity = student.person_id
        count = counts.get(identity, 0)
        best_dist = best_distance.get(identity)

        if count == 0:
            status = WindowStatus.NOT_SEEN
            dist_val = None
        elif count >= k:
            status = WindowStatus.MATCHED
            dist_val = best_dist
        else:
            status = WindowStatus.UNCERTAIN
            dist_val = best_dist

        results.append(
            StudentWindowResult(
                roll_no=student.roll_no,
                status=status,
                distance=dist_val,
            )
        )

    status_by_roll = {r.roll_no: r.status for r in results}
    assert status_by_roll["r1"] == WindowStatus.MATCHED
    assert status_by_roll["r2"] == WindowStatus.UNCERTAIN
    assert status_by_roll["r3"] == WindowStatus.NOT_SEEN


def test_window_payload_jsonable_encoder():
    payload = WindowResultPayload(
        window_name="manual",
        timestamp=datetime.utcnow(),
        results=[
            StudentWindowResult(
                roll_no="23P-3040",
                status=WindowStatus.MATCHED,
                distance=0.1,
            )
        ],
    )
    data = jsonable_encoder(payload)
    assert isinstance(data, dict)
    assert isinstance(data["timestamp"], str)
    assert data["results"][0]["status"] == "matched"
