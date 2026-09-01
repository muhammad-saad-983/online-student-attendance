from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app, roster, settings  # noqa: E402
from app.services.export import export_current_attendance_to_csv  # noqa: E402
from app.state import get_state  # noqa: E402


def test_export_helper_creates_csv(tmp_path: Path):
    state = get_state()
    state.reset()

    path = export_current_attendance_to_csv(settings, roster, state, tmp_path)
    assert path.exists()

    content = path.read_text().strip().splitlines()
    assert content[0] == "roll_no,name,early,mid,manual,final"
    assert len(content) >= 1


def test_export_csv_endpoint():
    client = TestClient(app)
    resp = client.get("/api/export/csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")
    body = resp.text.strip().splitlines()
    assert body[0].startswith("roll_no,name,early,mid,manual,final")
