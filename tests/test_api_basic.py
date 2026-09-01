from datetime import datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app, roster  # noqa: E402
from app.schemas import WindowStatus  # noqa: E402
from app.state import get_state  # noqa: E402


def get_client():
    return TestClient(app)


def reset_state():
    state = get_state()
    state.reset()
    return state


def test_get_index_renders_roster_name():
    reset_state()
    client = get_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Hassan" in resp.text


def test_trigger_window_sets_running_status():
    state = reset_state()
    client = get_client()

    trigger_resp = client.post("/api/trigger-window", headers={"HX-Request": "true"})
    assert trigger_resp.status_code == 200
    assert state.windows["manual"].status in {"running", "done"}


def test_trigger_window_with_body_sets_mid_running():
    state = reset_state()
    client = get_client()

    trigger_resp = client.post(
        "/api/trigger-window",
        json={"window_name": "mid"},
        headers={"HX-Request": "true"},
    )
    assert trigger_resp.status_code == 200
    assert state.windows["mid"].status in {"running", "done"}


def test_window_result_applies_and_marks_done_and_finalize():
    state = reset_state()
    client = get_client()

    payload = {
        "window_name": "manual",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": [
            {
                "roll_no": roster[0].roll_no,
                "status": "matched",
                "distance": 0.12,
            }
        ],
    }

    result_resp = client.post("/api/window-result", json=payload)
    assert result_resp.status_code == 200
    assert state.windows["manual"].status == "done"
    key = (roster[0].roll_no, "manual")
    assert state.per_student_window[key] == WindowStatus.MATCHED

    finalize_payload = {"timestamp": datetime.utcnow().isoformat() + "Z"}
    finalize_resp = client.post("/api/finalize", json=finalize_payload)
    assert finalize_resp.status_code == 200
    finalize_data = finalize_resp.json()
    assert len(finalize_data["decisions"]) == len(roster)
    assert finalize_data["policy_mode"]

    end_class_resp = client.post("/api/end-class", json=finalize_payload)
    assert end_class_resp.status_code == 200
    end_class_data = end_class_resp.json()
    assert len(end_class_data["decisions"]) == len(roster)
