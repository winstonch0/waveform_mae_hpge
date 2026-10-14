import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from dataset.constants import DatasetConstants
from dataset.dataset import MajoranaDataset


constants = DatasetConstants("dataset/stats.json")

files = sorted(
    Path("data/waveform_npz/MJD_Train_0").glob("*.npz")
)[:100]

if not files:
    raise RuntimeError("No NPZ files found")


# ------------------------------------------
# Expected gradient according to compute_stats
# ------------------------------------------

expected_all = []

for path in files:
    data = np.load(path)

    waveform = data["waveform"].astype(np.float64)
    tp0 = int(data.get("tp0", 1000))

    baseline = waveform[:tp0].mean()
    waveform = waveform - baseline

    gradient = np.gradient(waveform)

    gradient = (
        gradient - constants.GRADIENT_MEAN
    ) / constants.GRADIENT_STD

    expected_all.append(gradient)

expected = np.concatenate(expected_all)


# ------------------------------------------
# ACTUAL runtime pipeline from MajoranaDataset
# ------------------------------------------

dataset = MajoranaDataset(
    root="data/waveform_npz/MJD_Train_0",
    window_size=1,
    constants=constants,
    to_torch=True,
    shift_augmentation=0,
    use_second_deriv=False,
)

runtime_all = []

for i in range(min(100, len(dataset))):
    event = dataset[i]

    gradient = event["gradient"].numpy().reshape(-1)

    runtime_all.append(gradient)

runtime = np.concatenate(runtime_all)


# ------------------------------------------
# Compare
# ------------------------------------------

diff = runtime - expected

print("Files checked:", len(files))
print()

print("WAVEFORM_STD:", constants.WAVEFORM_STD)
print("GRADIENT_STD:", constants.GRADIENT_STD)

print("\nGradient normalized according to stats pipeline:")
print("  mean:", expected.mean())
print("  std: ", expected.std())

print("\nGradient produced by ACTUAL MajoranaDataset:")
print("  mean:", runtime.mean())
print("  std: ", runtime.std())

print("\nDifference:")
print("  mean absolute difference:", np.abs(diff).mean())
print("  max absolute difference: ", np.abs(diff).max())
print("  RMSE:                    ", np.sqrt(np.mean(diff**2)))

print("\nRuntime / expected std ratio:")
print(runtime.std() / expected.std())