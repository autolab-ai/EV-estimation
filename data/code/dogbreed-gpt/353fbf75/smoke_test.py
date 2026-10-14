"""Fast synthetic checks; does not read competition data or invoke a grader."""
import os
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset

from train import Tracking, fit_temperature, tracked_run, worker_tempdir


def main():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(24, 3))
    targets = np.tile(np.arange(3), 8)
    temperature = fit_temperature(logits, targets)
    assert 0.5 <= temperature <= 3.0
    probabilities = softmax(logits / temperature, axis=1)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    # Reproduce the long inherited temp path without running image extraction.
    cached_tempdir = tempfile.tempdir
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="dog-smoke-") as root:
        deep_path = os.path.join(root, "nested-" * 24)
        os.makedirs(deep_path)
        with patch.dict(os.environ, {key: deep_path for key in ("TMPDIR", "TEMP", "TMP")}):
            tempfile.tempdir = deep_path
            with worker_tempdir():
                assert len(tempfile.gettempdir()) < 70
                loader = DataLoader(TensorDataset(torch.arange(32)), batch_size=4, num_workers=8)
                actual = torch.cat([batch[0] for batch in loader])
                assert torch.equal(actual, torch.arange(32))
            assert tempfile.tempdir == deep_path
        tempfile.tempdir = cached_tempdir

    import mlflow
    with patch.object(mlflow, "start_run", side_effect=RuntimeError("synthetic tracking failure")):
        with tracked_run() as tracking:
            assert tracking.client is None
            tracking.log("log_metric", "smoke", 1.0)
    tracker = Tracking()
    with patch.object(mlflow, "log_metric", side_effect=RuntimeError("synthetic logging failure")):
        tracker.client = mlflow
        tracker.log("log_metric", "smoke", 1.0)
        assert tracker.client is None
    print("Smoke checks passed: calibration, eight-worker transfer, tracking fallback")


if __name__ == "__main__":
    main()
