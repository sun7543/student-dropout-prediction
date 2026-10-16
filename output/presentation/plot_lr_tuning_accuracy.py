import matplotlib.pyplot as plt
import numpy as np


# Results from new_notebook/6_hyperparameter_tuning.ipynb.
models = ["Baseline LR", "Tuned LR (C=0.01)"]
train_accuracy = [411 / 560 * 100, 413 / 560 * 100]
validation_accuracy = [79 / 120 * 100, 82 / 120 * 100]
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width / 2, train_accuracy, width, label="Train", color="tab:blue")
plt.bar(x + width / 2, validation_accuracy, width, label="Validation", color="tab:orange")
plt.xticks(x, models)
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Logistic Regression Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("output/presentation/lr-tuning-accuracy.png", dpi=180)
