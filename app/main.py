from datetime import datetime, timezone
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import structlog

from app.config import AppSettings, get_settings
from app.logging_config import configure_logging
from app.schemas import (
    BasicResponse,
    FinalDecision,
    FinalStatus,
    FinalizeRequest,
    FinalizeResponse,
    FinalizeStudentResult,
    TriggerWindowRequest,
    WindowResultPayload,
    WindowStatus,
)
from app.services.export import export_current_attendance_to_csv
from app.services.roster_service import load_roster
from app.services.policy import build_roster_view
from app.services.window_service import apply_window_result
from app.state import AttendanceState, WindowInfo, get_state

configure_logging()
audit_logger = structlog.get_logger("attendance")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "data" / "templates"
STATIC_DIR = BASE_DIR / "data" / "static"
ROSTER_PATH = BASE_DIR / "data" / "enrollment" / "roster.csv"

settings: AppSettings = get_settings()
roster = load_roster(ROSTER_PATH)

app = FastAPI(title="Face Attendance")

# Mount static files for assets like icons and images
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    state = get_state()
    roster_rows, summary = build_roster_view(settings, roster, state)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "settings": settings,
            "roster_rows": roster_rows,
            "summary": summary,
            "state": state,
        },
    )


def _finalize(timestamp: datetime) -> FinalizeResponse:
    state = get_state()
    if state.finalized and state.finalized_at:
        return FinalizeResponse(
            decisions=[
                FinalizeStudentResult(roll_no=roll_no, decision=decision)
                for roll_no, decision in state.final_decisions.items()
            ],
            finalized_at=state.finalized_at,
            policy_mode=settings.policy_mode,
        )

    decisions: List[FinalizeStudentResult] = []
    for student in roster:
        statuses = {
            "early": state.per_student_window.get((student.roll_no, "early")),
            "mid": state.per_student_window.get((student.roll_no, "mid")),
            "manual": state.per_student_window.get((student.roll_no, "manual")),
        }
        if WindowStatus.MATCHED in statuses.values():
            decision = FinalDecision.PRESENT
        else:
            decision = FinalDecision.ABSENT
        decisions.append(FinalizeStudentResult(roll_no=student.roll_no, decision=decision))
        state.final_decisions[student.roll_no] = decision

    state.finalized = True
    state.finalized_at = timestamp
    return FinalizeResponse(
        decisions=decisions,
        finalized_at=timestamp,
        policy_mode=settings.policy_mode,
    )


logger = logging.getLogger(__name__)


def run_window_and_apply(window_name: str) -> None:
    """
    Background task: spawn `python -m vision.run_window` as a subprocess.

    The subprocess:
      - Opens the webcam on its own main thread.
      - Runs the K-of-N recognition window.
      - Posts results to /api/window-result via HTTP.

    This function does NOT touch OpenCV or modify AttendanceState directly.
    """
    api_url = os.environ.get("FACE_ATTENDANCE_API_URL", "http://localhost:8000")

    cmd = [
        sys.executable,
        "-m",
        "vision.run_window",
        "--window-name",
        window_name,
        "--api-url",
        api_url,
    ]

    env = os.environ.copy()
    env.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

    logger.info("Starting vision window subprocess: %s", " ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info("Vision window subprocess finished (returncode=%s)", completed.returncode)
        if completed.stdout:
            logger.debug("vision.run_window stdout:\n%s", completed.stdout)
        if completed.stderr:
            logger.debug("vision.run_window stderr:\n%s", completed.stderr)
    except Exception as exc:
        logger.exception("Failed to run vision window subprocess: %s", exc)


@app.post("/api/trigger-window", response_class=HTMLResponse)
async def trigger_window(
    request: Request,
    background_tasks: BackgroundTasks,
    body: Optional[TriggerWindowRequest] = None,
):
    state = get_state()
    window_name = body.window_name if body is not None else "manual"

    if window_name not in ("early", "mid", "manual"):
        raise HTTPException(status_code=400, detail="Invalid window_name")

    info = state.windows.get(window_name)
    if info is None:
        info = WindowInfo()
        state.windows[window_name] = info

    info.status = "running"
    info.last_updated = datetime.now(timezone.utc)

    if not settings.demo_mode:
        background_tasks = background_tasks or BackgroundTasks()
        background_tasks.add_task(run_window_and_apply, window_name)

    return templates.TemplateResponse(
        "timeline.html",
        {
            "request": request,
            "settings": settings,
            "state": state,
        },
    )


@app.post("/api/window-result", response_model=BasicResponse)
async def window_result(payload: WindowResultPayload):
    state = get_state()
    apply_window_result(payload, state)
    return BasicResponse(ok=True, message="Window results recorded")


@app.post("/api/finalize", response_model=FinalizeResponse)
async def finalize(payload: FinalizeRequest):
    return _finalize(timestamp=payload.timestamp)


@app.post("/api/end-class", response_model=FinalizeResponse)
async def end_class(payload: FinalizeRequest):
    state = get_state()
    state.class_ended = True
    return _finalize(timestamp=payload.timestamp)


@app.post("/api/override-final")
async def override_final(
    roll_no: str = Form(...),
    final_status: FinalStatus = Form(...),
    reason: str = Form(""),
):
    """
    Manual override for a student's final status.
    """
    state = get_state()
    state.final_overrides[roll_no] = final_status
    if reason:
        state.override_reasons[roll_no] = reason

    label = f"Manual override: {roll_no} → {final_status.value.title()}"
    state.record_last_action(label=label)

    audit_logger.info(
        "manual_override",
        roll_no=roll_no,
        final_status=final_status.value,
        reason=reason or None,
    )

    return RedirectResponse(url="/", status_code=303)


@app.get("/api/export/csv")
async def export_csv():
    """
    Export the current attendance snapshot as a CSV and return it as a download.
    """
    state = get_state()
    exports_dir = Path("data") / "exports"
    path = export_current_attendance_to_csv(settings, roster, state, exports_dir)

    audit_logger.info("attendance_export_csv", path=str(path), filename=path.name)

    return FileResponse(
        path,
        media_type="text/csv",
        filename=path.name,
    )
