from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from app.schemas import FinalStatus, WindowStatus  # noqa: E402
from app.services.policy import build_roster_view  # noqa: E402
from app.state import get_state  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.services.roster_service import load_roster  # noqa: E402


def test_override_takes_precedence_over_policy():
    client = TestClient(app)
    state = get_state()
    state.reset()

    settings = get_settings()
    roll_no = "23P-3040"
    state.per_student_window[(roll_no, "early")] = WindowStatus.NOT_SEEN
    state.per_student_window[(roll_no, "mid")] = WindowStatus.NOT_SEEN

    roster = load_roster(Path("data/enrollment/roster.csv"))

    rows, summary = build_roster_view(settings, roster, state)
    row = next(r for r in rows if r.roll_no == roll_no)
    assert row.final == FinalStatus.ABSENT

    state.final_overrides[roll_no] = FinalStatus.PRESENT
    rows2, summary2 = build_roster_view(settings, roster, state)
    row2 = next(r for r in rows2 if r.roll_no == roll_no)
    assert row2.final == FinalStatus.PRESENT
    assert summary2["present"] >= summary["present"]


def test_last_action_recorded_on_override():
    client = TestClient(app)
    state = get_state()
    state.reset()

    resp = client.post(
        "/api/override-final",
        data={"roll_no": "23P-3040", "final_status": "present"},
        follow_redirects=False,
    )
    assert resp.status_code in (303, 307)
    assert state.last_action_label is not None
    assert "23P-3040" in state.last_action_label
