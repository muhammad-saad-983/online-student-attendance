from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.build_embeddings import compute_distributions_and_threshold  # noqa: E402


def test_threshold_computation_runs_and_returns_stats():
    identity_to_mean = {
        "id1": np.array([1.0, 0.0], dtype=np.float32),
        "id2": np.array([0.0, 1.0], dtype=np.float32),
    }
    identity_to_all = {
        "id1": [np.array([0.8, 0.2], dtype=np.float32)],
        "id2": [np.array([0.2, 0.8], dtype=np.float32)],
    }

    threshold, stats = compute_distributions_and_threshold(identity_to_mean, identity_to_all)
    assert isinstance(threshold, float)
    assert "genuine" in stats
    assert "impostor" in stats
    assert "chosen" in stats
    assert stats["genuine"]["count"] > 0
