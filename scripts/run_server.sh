#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m uvicorn app.main:app --reload
