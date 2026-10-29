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

OUTPUT_DIR = Path("reconstruction_evaluation")
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
    ckpt
    for ckpt in checkpoints
    if "val_loss=" in ckpt.name
]

if not scored:
    raise RuntimeError(
        "Could not find checkpoint containing val_loss in filename"
    )

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

constants = DatasetConstants("dataset/stats.json")

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

val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(SEED)

_, val_subset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator,
)

print(f"Full dataset: {len(dataset)}")
print(f"Validation:   {len(val_subset)}")
print(f"Evaluating:   {min(NUM_EVENTS, len(val_subset))}")
print()


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# Evaluate fixed validation events
# ------------------------------------------------------------

results = []

n_events = min(
    NUM_EVENTS,
    len(val_subset)
)

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
            device=device,
        ).reshape(1)

    # Deterministic mask for this event
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

    # True means visible in model mask.
    # We evaluate only hidden/masked windows.
    masked_windows = (
        (~mask[0])
        .cpu()
        .numpy()
    )

    masked_samples = np.repeat(
        masked_windows,
        WINDOW_SIZE
    )

    # --------------------------------------------------------
    # Reconstruction MSE on MASKED samples only
    # --------------------------------------------------------

    waveform_mse = np.mean(
        (
            waveform_true[masked_samples]
            - waveform_pred[masked_samples]
        ) ** 2
    )

    gradient_mse = np.mean(
        (
            gradient_true[masked_samples]
            - gradient_pred[masked_samples]
        ) ** 2
    )

    # Simple zero-prediction baselines.
    # Helpful for judging whether reconstruction is meaningful.
    waveform_zero_mse = np.mean(
        waveform_true[masked_samples] ** 2
    )

    gradient_zero_mse = np.mean(
        gradient_true[masked_samples] ** 2
    )

    original_dataset_idx = int(
        val_subset.indices[val_idx]
    )

    result = {
        "val_index": val_idx,
        "dataset_index": original_dataset_idx,
        "waveform_mse": float(waveform_mse),
        "gradient_mse": float(gradient_mse),
        "waveform_zero_baseline": float(
            waveform_zero_mse
        ),
        "gradient_zero_baseline": float(
            gradient_zero_mse
        ),
    }

    if "detector" in event:
        result["detector"] = int(
            torch.as_tensor(
                event["detector"]
            ).item()
        )

    if "run_number" in event:
        result["run_number"] = int(
            np.asarray(
                event["run_number"]
            ).item()
        )

    results.append(result)

    if (
        val_idx % 10 == 0
        or val_idx == n_events - 1
    ):
        print(
            f"{val_idx + 1}/{n_events} "
            f"waveform={waveform_mse:.4f} "
            f"gradient={gradient_mse:.4f}"
        )


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

df = pd.DataFrame(results)

csv_path = (
    OUTPUT_DIR
    / "reconstruction_metrics_100.csv"
)

df.to_csv(
    csv_path,
    index=False,
)

print()
print(f"Saved: {csv_path}")


# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------

def summarize(name):
    values = df[name]

    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"mean:    {values.mean():.6f}"
    )
    print(
        f"median:  {values.median():.6f}"
    )
    print(
        f"std:     {values.std():.6f}"
    )
    print(
        f"min:     {values.min():.6f}"
    )
    print(
        f"90th %:  {values.quantile(0.90):.6f}"
    )
    print(
        f"95th %:  {values.quantile(0.95):.6f}"
    )
    print(
        f"max:     {values.max():.6f}"
    )


print("\n============================")
print("100-EVENT RECONSTRUCTION")
print("============================")

summarize(
    "waveform_mse"
)

summarize(
    "gradient_mse"
)


# ------------------------------------------------------------
# Compare against zero-prediction baseline
# ------------------------------------------------------------

mean_gradient_model = (
    df["gradient_mse"].mean()
)

mean_gradient_zero = (
    df["gradient_zero_baseline"].mean()
)

mean_waveform_model = (
    df["waveform_mse"].mean()
)

mean_waveform_zero = (
    df["waveform_zero_baseline"].mean()
)

print("\n============================")
print("ZERO-PREDICTION BASELINE")
print("============================")

print(
    "\nGradient:"
)
print(
    f"  model MSE:  {mean_gradient_model:.6f}"
)
print(
    f"  zero MSE:   {mean_gradient_zero:.6f}"
)

if mean_gradient_zero > 0:
    improvement = (
        1
        - mean_gradient_model
        / mean_gradient_zero
    ) * 100

    print(
        f"  improvement over zero: "
        f"{improvement:.2f}%"
    )

print(
    "\nWaveform:"
)
print(
    f"  model MSE:  {mean_waveform_model:.6f}"
)
print(
    f"  zero MSE:   {mean_waveform_zero:.6f}"
)

if mean_waveform_zero > 0:
    improvement = (
        1
        - mean_waveform_model
        / mean_waveform_zero
    ) * 100

    print(
        f"  improvement over zero: "
        f"{improvement:.2f}%"
    )


# ------------------------------------------------------------
# Find representative events
# ------------------------------------------------------------

sorted_gradient = df.sort_values(
    "gradient_mse"
)

best = sorted_gradient.iloc[0]

worst = sorted_gradient.iloc[-1]

median_target = (
    df["gradient_mse"].median()
)

median_idx = (
    df["gradient_mse"]
    .sub(median_target)
    .abs()
    .idxmin()
)

median_event = df.loc[
    median_idx
]

print("\n============================")
print("REPRESENTATIVE EVENTS")
print("============================")

print(
    "\nBest gradient reconstruction:"
)
print(
    f"  val index: "
    f"{int(best['val_index'])}"
)
print(
    f"  gradient MSE: "
    f"{best['gradient_mse']:.6f}"
)

print(
    "\nMedian gradient reconstruction:"
)
print(
    f"  val index: "
    f"{int(median_event['val_index'])}"
)
print(
    f"  gradient MSE: "
    f"{median_event['gradient_mse']:.6f}"
)

print(
    "\nWorst gradient reconstruction:"
)
print(
    f"  val index: "
    f"{int(worst['val_index'])}"
)
print(
    f"  gradient MSE: "
    f"{worst['gradient_mse']:.6f}"
)


# ------------------------------------------------------------
# Histograms
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    df["gradient_mse"],
    bins=20,
)

plt.axvline(
    df["gradient_mse"].median(),
    linestyle="--",
    label="Median",
)

plt.xlabel(
    "Masked gradient reconstruction MSE"
)

plt.ylabel(
    "Number of validation events"
)

plt.title(
    "Gradient Reconstruction Error — 100 Validation Events"
)

plt.legend()
plt.tight_layout()

gradient_plot = (
    OUTPUT_DIR
    / "gradient_mse_distribution.png"
)

plt.savefig(
    gradient_plot,
    dpi=200,
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

plt.hist(
    df["waveform_mse"],
    bins=20,
)

plt.axvline(
    df["waveform_mse"].median(),
    linestyle="--",
    label="Median",
)

plt.xlabel(
    "Masked waveform reconstruction MSE"
)

plt.ylabel(
    "Number of validation events"
)

plt.title(
    "Waveform Reconstruction Error — 100 Validation Events"
)

plt.legend()
plt.tight_layout()

waveform_plot = (
    OUTPUT_DIR
    / "waveform_mse_distribution.png"
)

plt.savefig(
    waveform_plot,
    dpi=200,
)

plt.close()


print()
print(
    f"Saved: {gradient_plot}"
)
print(
    f"Saved: {waveform_plot}"
)
