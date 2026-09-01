# Face Attendance

## Requirements
- Python 3.7.x
- macOS with webcam access for real attendance capture

## Setup
```bash
python3.7 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout for enrollment
- Roster: `data/enrollment/roster.csv`
  ```csv
  roll_no,person_id,name
  23P-3040,23P-3040_Hassan,Hassan
  23P-3066,23P-3066_Saad,Saad Shafi
  ```
- Raw videos: `data/enrollment/raw/{person_id}/*.mp4`
  - Example: `data/enrollment/raw/23P-3040_Hassan/enroll1.mp4`

## One-shot enrollment pipeline
Once roster + videos are in place, run:
```bash
python -m vision.enroll_all
```
This runs, in order:
1) Extract faces to `data/enrollment/faces/{person_id}/`
2) Train CNN → `models/face_cnn.pt` + `models/identity_mapping.json`
3) Build embeddings + thresholds → `data/templates/embeddings/` and `data/config/thresholds.yaml`

Individual steps (if needed):
```bash
python -m vision.extract_faces
python -m vision.train
python -m vision.build_embeddings
```

## Running the app
```bash
./scripts/run_server.sh
# then open http://localhost:8000
```
- Demo mode: set `demo_mode: true` in `data/config/config.yaml` (UI only; no webcam). 
- Real mode: set `demo_mode: false` ("Take Attendance Now" spawns webcam subprocess).

## Taking attendance
- Click **Take Attendance Now** (manual window). The webcam runs, posts results, and the dashboard updates Early/Mid/Manual chips and Final status.
- Manual overrides: use the **Override** dropdown per student to mark Present/Absent/Review.

## Exporting attendance
- Button: **Export CSV** (in the Attendance summary card) → downloads `/api/export/csv`.
- Files are also saved under `data/exports/attendance_YYYYmmdd_HHMMSS.csv`.

## Project layout
- `app/` FastAPI app, UI, policy, overrides
- `vision/` Enrollment + training + embeddings + window recognizer
- `data/` Config, roster, templates, static assets
- `models/` Trained weights + identity mapping
- `logs/` (for JSON logs if redirected)
