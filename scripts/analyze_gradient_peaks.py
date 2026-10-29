import sys
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import random_split

from dataset.dataset import MajoranaDataset
from dataset.constants import DatasetConstants
from model.lightning_mae import LitMAEWaveform


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_ROOT = "data/waveform_npz"
WINDOW_SIZE = 10
TRAIN_VAL_SPLIT = 0.9
SEED = 42
NUM_EVENTS = 100

CHECKPOINT_DIR = (
    Path(os.environ["SCRATCH"])
    / "waveform_mae_hpge_data"
    / "checkpoints"
    / "shard0_1k_posfix"
)

OUTPUT_DIR = Path("gradient_peak_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Find best checkpoint
# ------------------------------------------------------------

checkpoints = list(CHECKPOINT_DIR.glob("*.ckpt"))

if not checkpoints:
    raise RuntimeError(
        f"No checkpoints found in {CHECKPOINT_DIR}"
    )


def extract_val_loss(path):
    match = re.search(
        r"val_loss=([0-9]+(?:\.[0-9]+)?)",
        path.name
    )

    if match:
        return float(match.group(1))

    return float("inf")


scored = [
    path for path in checkpoints
    if "val_loss=" in path.name
]

checkpoint = min(
    scored,
    key=extract_val_loss
)

print("Using checkpoint:")
print(checkpoint)
print()


# ------------------------------------------------------------
# Recreate validation split
# ------------------------------------------------------------

constants = DatasetConstants(
    "dataset/stats.json"
)

dataset = MajoranaDataset(
    root=DATA_ROOT,
    window_size=WINDOW_SIZE,
    constants=constants,
    to_torch=True,
    normalize_energy=False,
    pattern_filter="MJD_Train",
    shift_augmentation=0,
    use_second_deriv=False,
)

train_size = int(
    TRAIN_VAL_SPLIT * len(dataset)
)

val_size = (
    len(dataset) - train_size
)

generator = torch.Generator().manual_seed(
    SEED
)

_, val_subset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator,
)

n_events = min(
    NUM_EVENTS,
    len(val_subset)
)

print(f"Dataset size: {len(dataset)}")
print(f"Validation size: {len(val_subset)}")
print(f"Evaluating: {n_events}")
print()


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = LitMAEWaveform.load_from_checkpoint(
    checkpoint,
    map_location=device,
)

model = model.to(device)
model.eval()

print(f"Device: {device}")
print()


# ------------------------------------------------------------
# Evaluate peak reconstruction
# ------------------------------------------------------------

results = []

for val_idx in range(n_events):

    event = val_subset[val_idx]

    waveform = (
        event["waveform"]
        .unsqueeze(0)
        .to(device)
    )

    gradient = (
        event["gradient"]
        .unsqueeze(0)
        .to(device)
    )

    second_deriv = (
        event["second_deriv"]
        .unsqueeze(0)
        .to(device)
    )

    detector = event.get(
        "detector",
        None
    )

    if detector is not None:
        detector = torch.as_tensor(
            detector,
            device=device
        ).reshape(1)

    # Same deterministic masking scheme
    torch.manual_seed(
        SEED + val_idx
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED + val_idx
        )

    with torch.no_grad():
        (
            waveform_recon,
            gradient_recon,
            _,
            mask,
        ) = model.model(
            waveform,
            gradient,
            second_deriv,
            detector=detector,
        )

    gradient_true = (
        gradient[0]
        .cpu()
        .numpy()
        .reshape(-1)
    )

    gradient_pred = (
        gradient_recon[0]
        .cpu()
        .numpy()
        .reshape(-1)
    )

    waveform_true = (
        waveform[0]
        .cpu()
        .numpy()
        .reshape(-1)
    )

    waveform_pred = (
        waveform_recon[0]
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # mask=True means visible.
    # We care only about masked samples.
    masked_windows = (
        (~mask[0])
        .cpu()
        .numpy()
    )

    masked_samples = np.repeat(
        masked_windows,
        WINDOW_SIZE
    )

    masked_indices = np.flatnonzero(
        masked_samples
    )

    true_masked_grad = (
        gradient_true[masked_samples]
    )

    pred_masked_grad = (
        gradient_pred[masked_samples]
    )

    # --------------------------------------------------------
    # Standard reconstruction errors
    # --------------------------------------------------------

    gradient_mse = np.mean(
        (
            true_masked_grad
            - pred_masked_grad
        ) ** 2
    )

    waveform_mse = np.mean(
        (
            waveform_true[masked_samples]
            - waveform_pred[masked_samples]
        ) ** 2
    )

    # --------------------------------------------------------
    # Peak statistics
    # --------------------------------------------------------

    # Largest TRUE gradient among locations the model had to reconstruct
    local_true_peak_idx = np.argmax(
        np.abs(true_masked_grad)
    )

    true_peak_sample = int(
        masked_indices[
            local_true_peak_idx
        ]
    )

    true_peak_mag = float(
        np.abs(
            gradient_true[
                true_peak_sample
            ]
        )
    )

    # What did the model predict at that SAME physical location?
    pred_at_true_peak = float(
        np.abs(
            gradient_pred[
                true_peak_sample
            ]
        )
    )

    if true_peak_mag > 1e-12:
        peak_recovery_ratio = (
            pred_at_true_peak
            / true_peak_mag
        )
    else:
        peak_recovery_ratio = np.nan

    # Largest peak predicted anywhere in the masked region
    local_pred_peak_idx = np.argmax(
        np.abs(pred_masked_grad)
    )

    pred_peak_sample = int(
        masked_indices[
            local_pred_peak_idx
        ]
    )

    pred_peak_mag = float(
        np.abs(
            gradient_pred[
                pred_peak_sample
            ]
        )
    )

    peak_location_error = abs(
        pred_peak_sample
        - true_peak_sample
    )

    result = {
        "val_index": val_idx,
        "dataset_index": int(
            val_subset.indices[val_idx]
        ),
        "waveform_mse": float(
            waveform_mse
        ),
        "gradient_mse": float(
            gradient_mse
        ),
        "true_peak_mag": true_peak_mag,
        "pred_at_true_peak": pred_at_true_peak,
        "peak_recovery_ratio": peak_recovery_ratio,
        "pred_peak_mag": pred_peak_mag,
        "true_peak_sample": true_peak_sample,
        "pred_peak_sample": pred_peak_sample,
        "peak_location_error": peak_location_error,
    }

    if "run_number" in event:
        result["run_number"] = int(
            np.asarray(
                event["run_number"]
            ).item()
        )

    if "detector" in event:
        result["detector"] = int(
            torch.as_tensor(
                event["detector"]
            ).item()
        )

    results.append(result)

    if (
        val_idx % 10 == 0
        or val_idx == n_events - 1
    ):
        print(
            f"{val_idx + 1}/{n_events}: "
            f"peak={true_peak_mag:.2f}, "
            f"recovered={peak_recovery_ratio:.2f}, "
            f"grad_mse={gradient_mse:.3f}"
        )


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

df = pd.DataFrame(results)

csv_path = (
    OUTPUT_DIR
    / "gradient_peak_metrics.csv"
)

df.to_csv(
    csv_path,
    index=False,
)

print()
print(f"Saved {csv_path}")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n============================")
print("PEAK RECONSTRUCTION SUMMARY")
print("============================")

print(
    f"Median true peak magnitude: "
    f"{df['true_peak_mag'].median():.3f}"
)

print(
    f"Median peak recovery ratio: "
    f"{df['peak_recovery_ratio'].median():.3f}"
)

print(
    f"Mean peak recovery ratio: "
    f"{df['peak_recovery_ratio'].mean():.3f}"
)

print(
    f"Median peak location error: "
    f"{df['peak_location_error'].median():.1f} samples"
)

print(
    f"90th percentile location error: "
    f"{df['peak_location_error'].quantile(0.90):.1f} samples"
)

corr = df[
    [
        "true_peak_mag",
        "gradient_mse",
    ]
].corr().iloc[0, 1]

print(
    f"Correlation(true peak magnitude, gradient MSE): "
    f"{corr:.3f}"
)


# ------------------------------------------------------------
# Worst peak-recovery cases
# ------------------------------------------------------------

print("\n============================")
print("WORST PEAK RECOVERY")
print("============================")

cols = [
    "val_index",
    "gradient_mse",
    "true_peak_mag",
    "pred_at_true_peak",
    "peak_recovery_ratio",
    "peak_location_error",
]

print(
    df.sort_values(
        "peak_recovery_ratio"
    )[cols]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Plot 1:
# Gradient MSE vs true peak magnitude
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    df["true_peak_mag"],
    df["gradient_mse"],
    alpha=0.7,
)

plt.xlabel(
    "True maximum |gradient| in masked region"
)

plt.ylabel(
    "Masked gradient reconstruction MSE"
)

plt.title(
    "Gradient reconstruction error vs pulse sharpness"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "gradient_mse_vs_peak.png"
)

plt.savefig(
    path,
    dpi=200,
)

plt.close()

print(f"Saved {path}")


# ------------------------------------------------------------
# Plot 2:
# Peak recovery ratio vs true peak magnitude
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    df["true_peak_mag"],
    df["peak_recovery_ratio"],
    alpha=0.7,
)

plt.axhline(
    1.0,
    linestyle="--",
    label="Perfect amplitude recovery",
)

plt.xlabel(
    "True maximum |gradient| in masked region"
)

plt.ylabel(
    "Predicted / true gradient at true peak"
)

plt.title(
    "MAE recovery of sharp gradient peaks"
)

plt.legend()
plt.tight_layout()

path = (
    OUTPUT_DIR
    / "peak_recovery_vs_peak.png"
)

plt.savefig(
    path,
    dpi=200,
)

plt.close()

print(f"Saved {path}")


# ------------------------------------------------------------
# Plot 3:
# Peak location error
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    df["peak_location_error"],
    bins=20,
)

plt.xlabel(
    "Peak location error (samples)"
)

plt.ylabel(
    "Number of events"
)

plt.title(
    "Gradient peak localization error"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "peak_location_error.png"
)

plt.savefig(
    path,
    dpi=200,
)

plt.close()

print(f"Saved {path}")
