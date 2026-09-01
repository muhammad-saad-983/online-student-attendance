from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from pydantic import BaseModel


class AppSettings(BaseModel):
    demo_mode: bool
    windows: List[Dict[str, Any]]
    policy_mode: str
    k_of_n: Tuple[int, int]
    match_threshold: float
    manual_affects_final: bool
    end_class_face_fraction: float
    end_class_empty_seconds: int


CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "config" / "config.yaml"


def load_settings() -> AppSettings:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    return AppSettings(**raw)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return load_settings()
