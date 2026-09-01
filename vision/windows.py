from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi.encoders import jsonable_encoder
from PIL import Image
from torchvision import transforms
import httpx
import yaml

from app.config import get_settings
from app.models import Student
from app.schemas import StudentWindowResult, WindowResultPayload, WindowStatus
from app.services.roster_service import load_roster
from vision.model import FaceNet
from vision.train_config import TrainSettings, load_train_settings


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - float(np.dot(a_norm, b_norm)))


def load_identity_embeddings() -> Dict[str, np.ndarray]:
    embeddings_dir = Path("data/templates/embeddings")
    if not embeddings_dir.exists():
        return {}

    identity_to_vec: Dict[str, np.ndarray] = {}
    for npz_path in embeddings_dir.glob("*.npz"):
        data = np.load(npz_path)
        vec = data["embedding"]
        identity_to_vec[npz_path.stem] = vec.astype(np.float32)
    return identity_to_vec


def load_match_threshold() -> float:
    thresholds_path = Path("data/config/thresholds.yaml")
    if not thresholds_path.exists():
        return 0.4
    raw = yaml.safe_load(thresholds_path.read_text())
    return float(raw.get("match_threshold", 0.4))


def load_roster_mapping() -> Tuple[List[Student], Dict[str, Student]]:
    roster_path = Path("data/enrollment/roster.csv")
    roster = load_roster(roster_path)
    by_person_id: Dict[str, Student] = {}
    for student in roster:
        by_person_id[student.person_id] = student
    return roster, by_person_id


def get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return detector


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ]
    )


@dataclass
class RecognizerConfig:
    window_name: str
    api_url: str = "http://localhost:8000"
    camera_index: int = 0
    num_frames: Optional[int] = None


@dataclass
class FrameVotes:
    counts: Dict[str, int] = field(default_factory=dict)
    best_distance: Dict[str, float] = field(default_factory=dict)

    def register(self, identity: str, distance: float) -> None:
        self.counts[identity] = self.counts.get(identity, 0) + 1
        if identity not in self.best_distance or distance < self.best_distance[identity]:
            self.best_distance[identity] = distance


class WindowRecognizer:
    def __init__(self, config: RecognizerConfig):
        self.config = config
        self.app_settings = get_settings()
        self.train_settings: TrainSettings = load_train_settings()

        self.identity_to_vec = load_identity_embeddings()
        self.match_threshold = load_match_threshold()
        self.roster, self.by_person_id = load_roster_mapping()

        if not self.identity_to_vec:
            print("[window] No identity embeddings loaded; all students will be NOT_SEEN.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = max(len(self.identity_to_vec), 1)
        self.model = FaceNet(num_classes=num_classes, embedding_dim=self.train_settings.embedding_dim).to(self.device)

        weights_path = self.train_settings.models_dir / "face_cnn.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"[window] Model weights not found at {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.detector = get_face_detector()
        self.transform = get_eval_transform()

    def _infer_embedding(self, image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        h_img, w_img = image_bgr.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w_img, x + w)
        y1 = min(h_img, y + h)
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, emb = self.model(tensor)
        return emb.cpu().numpy()[0]

    def _detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        faces = list(faces)
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        return faces

    def _match_identity(self, emb: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.identity_to_vec:
            return None

        best_identity = None
        best_dist = float("inf")
        for identity, ref_vec in self.identity_to_vec.items():
            d = cosine_distance(emb, ref_vec)
            if d < best_dist:
                best_dist = d
                best_identity = identity

        if best_identity is None:
            return None

        if best_dist <= self.match_threshold:
            return best_identity, best_dist
        return None

    def run_window(self) -> WindowResultPayload:
        k, n_default = self.app_settings.k_of_n
        num_frames = self.config.num_frames or n_default

        votes = FrameVotes()

        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            print(f"[window] Failed to open camera index {self.config.camera_index}")
            now = datetime.now(timezone.utc)
            results = [
                StudentWindowResult(roll_no=student.roll_no, status=WindowStatus.NOT_SEEN, distance=None)
                for student in self.roster
            ]
            return WindowResultPayload(window_name=self.config.window_name, timestamp=now, results=results)

        try:
            frame_idx = 0
            while frame_idx < num_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                faces = self._detect_faces(frame)
                if not faces:
                    continue

                for bbox in faces[:1]:
                    emb = self._infer_embedding(frame, bbox)
                    if emb is None:
                        continue
                    match = self._match_identity(emb)
                    if match is None:
                        continue
                    identity, dist = match
                    votes.register(identity, dist)
        finally:
            cap.release()

        now = datetime.now(timezone.utc)
        results: List[StudentWindowResult] = []

        for student in self.roster:
            identity = student.person_id
            count = votes.counts.get(identity, 0)
            best_dist = votes.best_distance.get(identity)

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

        return WindowResultPayload(
            window_name=self.config.window_name,
            timestamp=now,
            results=results,
        )

    def post_results(self, payload: WindowResultPayload) -> None:
        url = self.config.api_url.rstrip("/") + "/api/window-result"
        with httpx.Client(timeout=10.0) as client:
            data = jsonable_encoder(payload)
            resp = client.post(url, json=data)
            resp.raise_for_status()
            print(f"[window] Posted results to {url}, status={resp.status_code}")
