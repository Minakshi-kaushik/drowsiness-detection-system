"""
Training Plots

Author: Minakshi Kaushik
"""

import matplotlib.pyplot as plt


def save_training_plot(history, save_path):

    plt.figure(figsize=(10, 5))

    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()
