from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.train_config import load_train_settings, TrainSettings  # noqa: E402
from vision.train import FacesDataset  # noqa: E402


def test_load_train_settings_types():
    settings = load_train_settings()
    assert isinstance(settings, TrainSettings)
    assert isinstance(settings.faces_root, Path)
    assert isinstance(settings.models_dir, Path)
    assert settings.batch_size > 0
    assert 0 <= settings.val_split <= 1


def test_faces_dataset_empty(tmp_path: Path):
    ds = FacesDataset(tmp_path)
    assert len(ds) == 0
    assert ds.identity_to_label == {}


def test_faces_dataset_with_images(tmp_path: Path):
    id_dir = tmp_path / "id1"
    id_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        img = Image.new("RGB", (112, 112), color=(idx * 40, 0, 0))
        img.save(id_dir / f"img_{idx}.jpg")

    ds = FacesDataset(tmp_path)
    assert len(ds) == 3
    assert ds.identity_to_label == {"id1": 0}
