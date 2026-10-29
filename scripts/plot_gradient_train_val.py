import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("training_metrics_posfix/train_gradient_loss.csv")
val = pd.read_csv("training_metrics_posfix/val_gradient_loss.csv")

plt.figure(figsize=(9, 5))

plt.plot(
    train["step"],
    train["value"],
    label="Training gradient loss",
    linewidth=2,
)

plt.plot(
    val["step"],
    val["value"],
    label="Validation gradient loss",
    linewidth=2,
)

plt.xlabel("Training step")
plt.ylabel("Gradient reconstruction loss")
plt.title("MAE Gradient Reconstruction — Train vs Validation")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig("gradient_train_vs_val.png", dpi=200)
print("Saved gradient_train_vs_val.png")
