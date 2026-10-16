import matplotlib.pyplot as plt
import numpy as np


# Accuracy from new_notebook/5_model_development.ipynb.
models = ["Logistic Regression", "Decision Tree", "Random Forest"]
train_accuracy = [411 / 560 * 100, 100, 100]
validation_accuracy = [79 / 120 * 100, 71 / 120 * 100, 73 / 120 * 100]
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, train_accuracy, width, label="Train", color="tab:blue")
plt.bar(x + width / 2, validation_accuracy, width, label="Validation", color="tab:orange")
plt.ylim(0, 100)
plt.xticks(x, models)
plt.ylabel("Accuracy (%)")
plt.title("Baseline Model Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("output/presentation/train-validation-model-accuracy.png", dpi=180)
