from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from vision.model import FaceNet
from vision.train_config import TrainSettings, load_train_settings


class FacesDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.root = root
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.label_to_identity: Dict[int, str] = {}
        self.identity_to_label: Dict[str, int] = {}

        self._scan()

    def _scan(self) -> None:
        label = 0
        for identity_dir in sorted(self.root.iterdir()):
            if not identity_dir.is_dir():
                continue
            identity_name = identity_dir.name
            image_paths = sorted(identity_dir.glob("*.jpg"))
            if not image_paths:
                continue

            self.identity_to_label[identity_name] = label
            self.label_to_identity[label] = identity_name

            for img_path in image_paths:
                self.samples.append((img_path, label))

            label += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ]
    )
    return train_tf, val_tf


def train_model(settings: TrainSettings) -> None:
    faces_root = settings.faces_root
    faces_root.mkdir(parents=True, exist_ok=True)
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = build_transforms()
    full_dataset = FacesDataset(faces_root, transform=None)

    if len(full_dataset.identity_to_label) < 1:
        print("[train] No identities found under", faces_root)
        return

    if len(full_dataset.identity_to_label) == 1:
        print("[train] Only one identity found; training will still run but classification is trivial.")

    n_total = len(full_dataset)
    n_val = int(n_total * settings.val_split)
    n_train = max(n_total - n_val, 1)
    if n_val == 0 and n_total > 1:
        n_val = 1
        n_train = n_total - 1

    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    train_dataset.dataset.transform = train_tf
    val_dataset.dataset.transform = val_tf

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(full_dataset.identity_to_label)
    model = FaceNet(num_classes=num_classes, embedding_dim=settings.embedding_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(settings.num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= max(total, 1)
        train_acc = correct / max(total, 1) if total > 0 else 0.0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits, _ = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(val_total, 1) if val_total > 0 else 1
        val_acc = val_correct / max(val_total, 1) if val_total > 0 else 0.0

        print(
            f"[train] Epoch {epoch+1}/{settings.num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_total > 0 and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()

    model_path = models_dir / "face_cnn.pt"
    torch.save(best_state, model_path)
    print(f"[train] Saved best model to {model_path}")

    mapping_path = models_dir / "identity_mapping.json"
    mapping = {
        "label_to_identity": full_dataset.label_to_identity,
        "identity_to_label": full_dataset.identity_to_label,
    }
    mapping_path.write_text(json.dumps(mapping, indent=2))
    print(f"[train] Saved identity mapping to {mapping_path}")


def main() -> None:
    settings = load_train_settings()
    train_model(settings)


if __name__ == "__main__":
    main()
