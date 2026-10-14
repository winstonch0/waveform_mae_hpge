from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


metrics_dir = Path("training_metrics")

possible_train = [
    "train_loss_epoch.csv",
    "train_loss.csv",
]

possible_val = [
    "val_loss.csv",
    "val_loss_epoch.csv",
]


def find_file(names):
    for name in names:
        path = metrics_dir / name
        if path.exists():
            return path
    return None


train_file = find_file(possible_train)
val_file = find_file(possible_val)

if train_file is None:
    print("Could not find train loss CSV")
else:
    train = pd.read_csv(train_file)

    plt.figure()
    plt.plot(train["step"], train["value"], marker="o", label="train")

    if val_file is not None:
        val = pd.read_csv(val_file)
        plt.plot(val["step"], val["value"], marker="o", label="validation")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Majorana MAE - Shard 0 Smoke Run")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=200)

    print("Saved training_curve.png")
