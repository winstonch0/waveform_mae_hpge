from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("training_metrics_large14")
OUT = Path("training_plots_large14")
OUT.mkdir(exist_ok=True)


def load(name):
    df = pd.read_csv(ROOT / f"{name}.csv")

    cols = {c.lower(): c for c in df.columns}

    step_col = cols.get("step")
    value_col = cols.get("value")

    if step_col is None or value_col is None:
        raise RuntimeError(
            f"Unexpected columns in {name}: {list(df.columns)}"
        )

    return df[step_col], df[value_col]


def make_plot(train_name, val_name, title, filename):
    train_step, train_value = load(train_name)
    val_step, val_value = load(val_name)

    plt.figure(figsize=(9, 5))
    plt.plot(train_step, train_value, label="train")
    plt.plot(val_step, val_value, marker="o", label="validation")

    plt.xlabel("Training step")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    path = OUT / filename
    plt.savefig(path, dpi=180)
    plt.close()

    print("Saved:", path)


make_plot(
    "train_loss",
    "val_loss",
    "Total MAE loss",
    "total_loss.png",
)

make_plot(
    "train_waveform_loss",
    "val_waveform_loss",
    "Waveform reconstruction loss",
    "waveform_loss.png",
)

make_plot(
    "train_gradient_loss",
    "val_gradient_loss",
    "Gradient reconstruction loss",
    "gradient_loss.png",
)
