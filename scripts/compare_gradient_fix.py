from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ORIGINAL = Path("training_metrics_1k")
FIXED = Path("training_metrics_gradfix")


def load(folder, name):
    df = pd.read_csv(folder / f"{name}.csv")
    return df


metrics = [
    ("val_loss", "Total validation loss"),
    ("val_waveform_loss", "Waveform validation loss"),
    ("val_gradient_loss", "Gradient validation loss"),
]

for metric, title in metrics:
    original = load(ORIGINAL, metric)
    fixed = load(FIXED, metric)

    plt.figure(figsize=(8, 5))

    plt.plot(
        original["step"],
        original["value"],
        label="Original"
    )

    plt.plot(
        fixed["step"],
        fixed["value"],
        label="Gradient normalization fixed"
    )

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    output = f"{metric}_original_vs_gradfix.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved {output}")
