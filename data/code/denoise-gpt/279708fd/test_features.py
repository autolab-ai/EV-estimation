"""Fast input-pipeline checks; no dataset scoring or full training run."""
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import grey_closing

from train import (
    BASE_CH, BATCH_SIZE, PATCH, EPOCHS, STEPS_PER_EPOCH,
    UNet, configure_execution, input_features, predict, sample_batch,
)


def main():
    device = configure_execution()
    assert device.type == "cpu"
    assert not torch.cuda.is_available()
    assert EPOCHS * STEPS_PER_EPOCH == 2000
    rng = np.random.default_rng(123)
    image = rng.random((147, 169), dtype=np.float32)
    features = input_features(image)
    assert features.shape == (2, 147, 169)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(features[0], image)
    expected = np.clip(
        image / np.maximum(grey_closing(image, size=(31, 31), mode="reflect"), 1 / 255),
        0, 1,
    )
    np.testing.assert_array_equal(features[1], expected)
    for value in (0.0, 1.0):
        flat = np.full((35, 37), value, dtype=np.float32)
        assert np.isfinite(input_features(flat)).all()

    # Using the raw image as target verifies identical crops and both flips.
    bx, by = sample_batch([features], [image], rng, BATCH_SIZE, PATCH)
    assert bx.shape == (BATCH_SIZE, 2, PATCH, PATCH)
    assert by.shape == (BATCH_SIZE, 1, PATCH, PATCH)
    torch.testing.assert_close(bx[:, :1], by, rtol=0, atol=0)

    # Replay RNG choices to check that the normalized channel is also aligned.
    replay = np.random.default_rng(123)
    replay.random((147, 169), dtype=np.float32)
    for index in range(BATCH_SIZE):
        replay.integers(1)
        row = replay.integers(image.shape[0] - PATCH + 1)
        col = replay.integers(image.shape[1] - PATCH + 1)
        expected_crop = features[:, row:row + PATCH, col:col + PATCH]
        if replay.random() < 0.5:
            expected_crop = expected_crop[:, :, ::-1]
        if replay.random() < 0.5:
            expected_crop = expected_crop[:, ::-1, :]
        np.testing.assert_array_equal(bx[index].numpy(), expected_crop)

    model = UNet(BASE_CH).to(device).to(memory_format=torch.channels_last)
    assert model.e1[0].in_channels == 2
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    bx = bx.to(device).to(memory_format=torch.channels_last)
    by = by.to(device).to(memory_format=torch.channels_last)
    start = time.monotonic()
    loss = F.mse_loss(model(bx), by)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"One synthetic training update completed in {time.monotonic() - start:.2f}s", flush=True)
    model.eval()
    output = predict(model, features, device, amp=False)
    assert output.shape == image.shape
    assert np.isfinite(output).all()
    assert ((output >= 0) & (output <= 1)).all()
    print("Feature, crop, CPU training and padded inference checks passed", flush=True)


if __name__ == "__main__":
    main()
