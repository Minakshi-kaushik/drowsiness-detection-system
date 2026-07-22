import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from inference.detect_sleep import SleepDetector


class SleepDetectorIntegrationTests(unittest.TestCase):
    def test_detect_uses_tensorflow_model_when_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "eye_state_model.keras")

            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(64, 64, 3)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(2, activation="softmax"),
                ]
            )
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            model.save(model_path)

            detector = SleepDetector(model_path=model_path)

            face_crop = np.zeros((120, 120, 3), dtype=np.uint8)
            face_crop[:, :] = (255, 255, 255)

            landmarks = []
            for i in range(468):
                landmarks.append((10 + (i % 10), 10 + (i // 10) % 10))

            result = detector.detect(landmarks, face_crop=face_crop)

            self.assertIn(result["status"], {"Awake", "Sleeping"})
            self.assertEqual(result["source"], "model")
            self.assertIn("ear", result)

    def test_detect_falls_back_to_ear_when_model_is_missing(self):
        detector = SleepDetector(model_path="models/eye/does-not-exist.keras")

        landmarks = []
        for i in range(468):
            landmarks.append((10 + (i % 10), 10 + (i // 10) % 10))

        result = detector.detect(landmarks, face_crop=None)

        self.assertEqual(result["source"], "ear")
        self.assertIn(result["status"], {"Awake", "Sleeping"})
        self.assertIn("ear", result)


if __name__ == "__main__":
    unittest.main()
