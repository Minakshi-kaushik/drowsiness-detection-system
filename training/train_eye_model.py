"""
Train Eye State CNN

Author: Minakshi Kaushik
"""

import os

import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)

from training.model import build_model
from training.plots import save_training_plot


# ==========================================================
# Configuration
# ==========================================================

IMAGE_SIZE = (64, 64)

BATCH_SIZE = 64

EPOCHS = 25

TRAIN_DIR = "datasets/train"

VAL_DIR = "datasets/val"

MODEL_DIR = "models/eye"

MODEL_PATH = os.path.join(MODEL_DIR, "eye_state_model.keras")


# ==========================================================
# Main
# ==========================================================


def main():

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_dataset.class_names

    print()

    print("Classes:", class_names)

    print()

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as file:
        for name in class_names:
            file.write(name + "\n")

    # ==========================================================
    # Data Augmentation
    # ==========================================================

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ]
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.map(
        lambda x, y: (
            normalization(data_augmentation(x)),
            y,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    validation_dataset = validation_dataset.map(
        lambda x, y: (
            normalization(x),
            y,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    train_dataset = train_dataset.cache().prefetch(AUTOTUNE)

    validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)

    print()

    print("=" * 60)

    print("Building CNN")

    print("=" * 60)

    model = build_model()

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        CSVLogger(
            os.path.join(
                MODEL_DIR,
                "training_log.csv",
            )
        ),
    ]

    print()

    print("=" * 60)

    print("Training Started")

    print("=" * 60)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print()

    print("=" * 60)

    print("Saving Model")

    print("=" * 60)

    model.save(MODEL_PATH)

    save_training_plot(
        history,
        os.path.join(
            MODEL_DIR,
            "training_history.png",
        ),
    )

    print()

    print("Training Completed Successfully")

    print()

    print("Saved Model:")

    print(MODEL_PATH)

    print()

    print("Training Graph Saved.")

    print()

    print("=" * 60)


if __name__ == "__main__":
    main()
