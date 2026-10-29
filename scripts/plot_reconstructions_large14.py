import sys
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import random_split

from dataset.dataset import MajoranaDataset
from dataset.constants import DatasetConstants
from model.lightning_mae import LitMAEWaveform


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_ROOT = (
    Path(os.environ["SCRATCH"])
    / "waveform_mae_hpge_data"
    / "large_npz"
    / "train"
)
WINDOW_SIZE = 10
TRAIN_VAL_SPLIT = 0.98
SEED = 42

CHECKPOINT_DIR = (
    Path(os.environ["SCRATCH"])
    / "waveform_mae_hpge_data"
    / "checkpoints"
    / "large14_5ep"
)

OUTPUT_DIR = Path("reconstruction_plots_large14")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Which validation events to plot
VAL_EVENT_INDICES = [0, 1, 2]
#VAL_EVENT_INDICES = [90, 27, 8]  # Which validation events to plot


# ------------------------------------------------------------
# Find best checkpoint automatically
# ------------------------------------------------------------

checkpoints = list(CHECKPOINT_DIR.glob("*.ckpt"))

if not checkpoints:
    raise RuntimeError(
        f"No .ckpt files found in {CHECKPOINT_DIR}"
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
    ckpt for ckpt in checkpoints
    if "val_loss=" in ckpt.name
]

if scored:
    checkpoint = min(scored, key=extract_val_loss)
else:
    checkpoint = CHECKPOINT_DIR / "last.ckpt"

print(f"Using checkpoint:\n{checkpoint}\n")


# ------------------------------------------------------------
# Recreate validation split exactly as training did
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

train_size = int(TRAIN_VAL_SPLIT * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(SEED)

train_subset, val_subset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator,
)

print(f"Dataset size: {len(dataset)}")
print(f"Train size:   {len(train_subset)}")
print(f"Val size:     {len(val_subset)}")


# ------------------------------------------------------------
# Load trained model
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
print(f"Mask ratio: {model.hparams.mask_ratio}")


# ------------------------------------------------------------
# Helper: shade consecutive masked windows
# ------------------------------------------------------------

def shade_masked_regions(ax, masked_windows, window_size):
    """
    masked_windows: boolean array of shape (num_windows,)
                    True means this window was MASKED.
    """

    start = None

    for i, masked in enumerate(masked_windows):
        if masked and start is None:
            start = i

        is_last = i == len(masked_windows) - 1

        if start is not None:
            if (not masked) or is_last:
                end = i if not masked else i + 1

                ax.axvspan(
                    start * window_size,
                    end * window_size,
                    alpha=0.08,
                )

                start = None


# ------------------------------------------------------------
# Reconstruct selected validation events
# ------------------------------------------------------------

for plot_num, val_idx in enumerate(VAL_EVENT_INDICES):

    event = val_subset[val_idx]

    waveform = event["waveform"].unsqueeze(0).to(device)
    gradient = event["gradient"].unsqueeze(0).to(device)
    second_deriv = event["second_deriv"].unsqueeze(0).to(device)

    detector = event.get("detector", None)

    if detector is not None:
        detector = torch.as_tensor(
            detector,
            device=device
        ).reshape(1)

    # Fix random masking so plots are reproducible
    torch.manual_seed(SEED + val_idx)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + val_idx)

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

    # --------------------------------------------------------
    # Flatten 380 x 10 back to 3800 samples
    # --------------------------------------------------------

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

    # mask=True means VISIBLE
    # so ~mask means MASKED
    masked_windows = (
        (~mask[0])
        .cpu()
        .numpy()
    )

    # Convert token mask to individual waveform samples
    masked_samples = np.repeat(
        masked_windows,
        WINDOW_SIZE
    )

    # Only display model predictions on masked samples.
    # Predictions at visible locations are not part of the MAE loss
    # and are therefore not meaningful reconstruction outputs.
    waveform_pred_masked = np.full_like(
        waveform_pred,
        np.nan
    )

    gradient_pred_masked = np.full_like(
        gradient_pred,
        np.nan
    )

    waveform_pred_masked[masked_samples] = (
        waveform_pred[masked_samples]
    )

    gradient_pred_masked[masked_samples] = (
        gradient_pred[masked_samples]
    )

    # --------------------------------------------------------
    # Compute reconstruction errors ONLY on masked samples
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

    waveform_error = np.full_like(
        waveform_true,
        np.nan
    )

    gradient_error = np.full_like(
        gradient_true,
        np.nan
    )

    waveform_error[masked_samples] = np.abs(
        waveform_true[masked_samples]
        - waveform_pred[masked_samples]
    )

    gradient_error[masked_samples] = np.abs(
        gradient_true[masked_samples]
        - gradient_pred[masked_samples]
    )

    print(
        f"\nValidation event {val_idx}:"
        f"\n  masked waveform MSE = {waveform_mse:.6f}"
        f"\n  masked gradient MSE = {gradient_mse:.6f}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    x = np.arange(len(waveform_true))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 11),
        sharex=True,
    )

    # Waveform reconstruction
    axes[0].plot(
        x,
        waveform_true,
        label="True waveform",
        linewidth=1.5,
    )

    axes[0].plot(
        x,
        waveform_pred_masked,
        label="MAE reconstruction (masked only)",
        linewidth=1.2,
    )

    shade_masked_regions(
        axes[0],
        masked_windows,
        WINDOW_SIZE,
    )

    axes[0].set_ylabel("Normalized amplitude")
    axes[0].set_title(
        f"Validation event {val_idx} — waveform reconstruction"
    )
    axes[0].legend()

    # Gradient reconstruction
    axes[1].plot(
        x,
        gradient_true,
        label="True gradient",
        linewidth=1.2,
    )

    axes[1].plot(
        x,
        gradient_pred_masked,
        label="MAE reconstruction (masked only)",
        linewidth=1.0,
    )

    shade_masked_regions(
        axes[1],
        masked_windows,
        WINDOW_SIZE,
    )

    axes[1].set_ylabel("Normalized gradient")
    axes[1].set_title(
        f"Gradient reconstruction — masked MSE = {gradient_mse:.4f}"
    )
    axes[1].legend()

    # Waveform error
    axes[2].plot(
        x,
        waveform_error,
        linewidth=1.0,
    )

    shade_masked_regions(
        axes[2],
        masked_windows,
        WINDOW_SIZE,
    )

    axes[2].set_ylabel("|error|")
    axes[2].set_title(
        f"Waveform absolute error — masked MSE = {waveform_mse:.4f}"
    )

    # Gradient error
    axes[3].plot(
        x,
        gradient_error,
        linewidth=1.0,
    )

    shade_masked_regions(
        axes[3],
        masked_windows,
        WINDOW_SIZE,
    )

    axes[3].set_ylabel("|error|")
    axes[3].set_xlabel("Waveform sample")
    axes[3].set_title("Gradient absolute error")

    fig.suptitle(
        "Majorana MAE masked reconstruction\n"
        "Shaded regions were hidden from the encoder",
        fontsize=14,
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / f"reconstruction_val_event_{val_idx}.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  saved: {output}")
