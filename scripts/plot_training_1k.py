from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

metrics_dir = Path("training_metrics_1k")


def find_metric(names):
    for name in names:
        path = metrics_dir / name
        if path.exists():
            print(f"Using {path}")
            return path
    return None


train_file = find_metric([
    "train_loss_epoch.csv",
    "train_loss.csv",
])

val_file = find_metric([
    "val_loss.csv",
    "val_loss_epoch.csv",
])

if train_file is None:
    raise FileNotFoundError("Could not find training-loss CSV")

train = pd.read_csv(train_file)

plt.figure(figsize=(9, 5))

plt.plot(
    train["step"],
    train["value"],
    marker="o",
    label="Training loss",
)

if val_file is not None:
    val = pd.read_csv(val_file)

    plt.plot(
        val["step"],
        val["value"],
        marker="o",
        label="Validation loss",
    )

plt.xlabel("Training step")
plt.ylabel("Reconstruction loss")
plt.title("Majorana MAE — 30 Epoch Development Run")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig("training_curve_1k.png", dpi=200)
plt.savefig("training_curve_1k.pdf")

print("Saved:")
print("  training_curve_1k.png")
print("  training_curve_1k.pdf")
