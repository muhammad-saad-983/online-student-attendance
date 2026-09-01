from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from vision.model import FaceNet
from vision.train_config import TrainSettings, load_train_settings
from vision.train import FacesDataset


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two 1D vectors (0 = identical, ~2 = opposite)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - float(np.dot(a_norm, b_norm)))


@dataclass
class EmbeddingsResult:
    identity_to_mean: Dict[str, np.ndarray]
    identity_to_all: Dict[str, List[np.ndarray]]


def build_identity_embeddings(settings: TrainSettings) -> EmbeddingsResult:
    faces_root = settings.faces_root
    models_dir = settings.models_dir

    faces_root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    eval_tf = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ]
    )
    dataset = FacesDataset(faces_root, transform=eval_tf)

    if len(dataset) == 0 or len(dataset.identity_to_label) < 1:
        print("[embeddings] No faces/identities found under", faces_root)
        return EmbeddingsResult(identity_to_mean={}, identity_to_all={})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(dataset.identity_to_label)
    model = FaceNet(num_classes=num_classes, embedding_dim=settings.embedding_dim).to(device)

    weights_path = models_dir / "face_cnn.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"[embeddings] Model weights not found at {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=settings.num_workers,
    )

    identity_to_all: Dict[str, List[np.ndarray]] = {identity: [] for identity in dataset.identity_to_label.keys()}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _, emb = model(images)
            emb_np = emb.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for e_vec, label in zip(emb_np, labels_np):
                identity = dataset.label_to_identity[int(label)]
                identity_to_all[identity].append(e_vec)

    identity_to_mean: Dict[str, np.ndarray] = {}
    for identity, emb_list in identity_to_all.items():
        if not emb_list:
            continue
        stack = np.stack(emb_list, axis=0)
        mean_vec = stack.mean(axis=0)
        identity_to_mean[identity] = mean_vec

    return EmbeddingsResult(identity_to_mean=identity_to_mean, identity_to_all=identity_to_all)


def save_identity_embeddings(identity_to_mean: Dict[str, np.ndarray]) -> None:
    embeddings_dir = Path("data/templates/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    for identity, vec in identity_to_mean.items():
        out_path = embeddings_dir / f"{identity}.npz"
        np.savez(out_path, embedding=vec.astype(np.float32))
        print(f"[embeddings] Saved embedding for {identity} -> {out_path}")


def compute_distributions_and_threshold(
    identity_to_mean: Dict[str, np.ndarray],
    identity_to_all: Dict[str, List[np.ndarray]],
) -> Tuple[float, Dict]:
    genuine_distances: List[float] = []
    impostor_distances: List[float] = []

    for identity, mean_vec in identity_to_mean.items():
        samples = identity_to_all.get(identity, [])
        for emb in samples:
            emb_arr = np.asarray(emb, dtype=np.float32)
            d = cosine_distance(mean_vec, emb_arr)
            genuine_distances.append(d)

    identities = list(identity_to_mean.keys())
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            id_i = identities[i]
            id_j = identities[j]
            d = cosine_distance(identity_to_mean[id_i], identity_to_mean[id_j])
            impostor_distances.append(d)

    stats = {}

    def make_stats(values: List[float]) -> Dict:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        arr = np.array(values, dtype=np.float32)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    stats["genuine"] = make_stats(genuine_distances)
    stats["impostor"] = make_stats(impostor_distances)

    default_threshold = 0.4
    strategy = "default_0.4"

    if stats["genuine"]["count"] > 0 and stats["impostor"]["count"] > 0:
        gen_max = stats["genuine"]["max"]
        imp_min = stats["impostor"]["min"]

        if gen_max < imp_min:
            threshold = (gen_max + imp_min) / 2.0
            strategy = "midpoint_between_gen_max_and_imp_min"
        else:
            threshold = float(imp_min * 0.9)
            strategy = "fallback_imp_min_0.9"
    elif stats["impostor"]["count"] > 0:
        threshold = float(stats["impostor"]["min"] * 0.9)
        strategy = "imp_only_0.9_min"
    elif stats["genuine"]["count"] > 0:
        threshold = float(stats["genuine"]["max"] * 1.1)
        strategy = "gen_only_1.1_max"
    else:
        threshold = default_threshold

    stats["chosen"] = {
        "strategy": strategy,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

    return float(threshold), stats


def write_thresholds_yaml(match_threshold: float, stats: Dict) -> None:
    thresholds_path = Path("data/config/thresholds.yaml")
    payload = {
        "distance_metric": "cosine",
        "match_threshold": float(match_threshold),
        "genuine": stats.get("genuine", {}),
        "impostor": stats.get("impostor", {}),
        "chosen": stats.get("chosen", {}),
    }
    thresholds_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"[embeddings] Wrote thresholds to {thresholds_path}")


def run_build_embeddings() -> None:
    settings = load_train_settings()
    result = build_identity_embeddings(settings)

    if not result.identity_to_mean:
        print("[embeddings] No embeddings to save (no identities).")
        return

    save_identity_embeddings(result.identity_to_mean)

    threshold, stats = compute_distributions_and_threshold(
        result.identity_to_mean,
        result.identity_to_all,
    )
    print(f"[embeddings] Selected match_threshold={threshold:.4f} using {stats['chosen']['strategy']}")

    write_thresholds_yaml(threshold, stats)


def main() -> None:
    run_build_embeddings()


if __name__ == "__main__":
    main()
