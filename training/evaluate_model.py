import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ==========================================================
# Configuration
# ==========================================================

IMAGE_SIZE = (64, 64)

BATCH_SIZE = 64

TEST_DIR = "datasets/test"

MODEL_PATH = "models/eye/eye_state_model.keras"

OUTPUT_DIR = "models/eye"

# ==========================================================
# Load Dataset
# ==========================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
)

class_names = test_dataset.class_names

normalization = tf.keras.layers.Rescaling(1.0 / 255)

test_dataset = test_dataset.map(lambda x, y: (normalization(x), y))

# ==========================================================
# Load Model
# ==========================================================

print("\nLoading trained model...\n")

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================================
# Evaluate
# ==========================================================

loss, accuracy = model.evaluate(test_dataset)

print("\n" + "=" * 60)
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")
print("=" * 60)

# ==========================================================
# Predictions
# ==========================================================

y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])

predictions = model.predict(test_dataset)

y_pred = np.argmax(predictions, axis=1)

# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
)

print(report)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt",
    ),
    "w",
) as file:
    file.write(report)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

disp.plot(cmap="Blues")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png",
    )
)

plt.close()

print("\nConfusion Matrix Saved.")
print("Classification Report Saved.")
print("\nEvaluation Completed Successfully.\n")
