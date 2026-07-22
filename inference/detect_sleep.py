"""
Sleep Detection Module
----------------------
Uses Eye Aspect Ratio (EAR) and, when available, a trained TensorFlow
model to determine whether a person is awake or sleeping.
"""

import math
import os

import cv2
import numpy as np

from config.settings import EYE_MODEL
from utils.logger import logger

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - defensive fallback for minimal environments
    tf = None


class SleepDetector:
    # MediaPipe FaceMesh landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    CLASS_NAMES = ["awake", "sleepy"]

    def __init__(self, ear_threshold=0.23, consecutive_frames=20, model_path=None):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.model_path = model_path or EYE_MODEL
        self.model = self._load_model()

    def _load_model(self):
        if tf is None:
            logger.warning(
                "TensorFlow is not available. Falling back to EAR-based detection."
            )
            return None

        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning(
                "Eye-state model not found at %s. Falling back to EAR-based detection.",
                self.model_path,
            )
            return None

        try:
            model = tf.keras.models.load_model(self.model_path)
            logger.info("Loaded eye-state model from %s", self.model_path)
            return model
        except Exception as exc:  # pragma: no cover - depends on runtime model format
            logger.warning("Unable to load eye-state model: %s", exc)
            return None

    def distance(self, p1, p2):
        return math.dist(p1, p2)

    def eye_aspect_ratio(self, eye):
        A = self.distance(eye[1], eye[5])
        B = self.distance(eye[2], eye[4])
        C = self.distance(eye[0], eye[3])

        if C == 0:
            return 0

        return (A + B) / (2.0 * C)

    def _extract_eye_image(self, face_crop, landmarks, eye_indices):
        if face_crop is None or face_crop.size == 0:
            return None

        eye_points = [landmarks[i] for i in eye_indices if i < len(landmarks)]
        if len(eye_points) < 3:
            return None

        xs = [point[0] for point in eye_points]
        ys = [point[1] for point in eye_points]

        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(face_crop.shape[1], max(xs))
        y2 = min(face_crop.shape[0], max(ys))

        padding = max(6, int((x2 - x1 + y2 - y1) * 0.15))
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(face_crop.shape[1], x2 + padding)
        y2 = min(face_crop.shape[0], y2 + padding)

        eye_image = face_crop[y1:y2, x1:x2]
        if eye_image.size == 0:
            return None

        return cv2.resize(eye_image, (64, 64))

    def _predict_with_model(self, face_crop, landmarks):
        if self.model is None or face_crop is None:
            return None

        predictions = []
        for eye_indices in (self.LEFT_EYE, self.RIGHT_EYE):
            eye_image = self._extract_eye_image(face_crop, landmarks, eye_indices)
            if eye_image is None:
                continue

            normalized = eye_image.astype("float32") / 255.0
            expanded = np.expand_dims(normalized, axis=0)
            prediction = self.model.predict(expanded, verbose=0)
            predicted_index = int(np.argmax(prediction[0]))
            predicted_label = self.CLASS_NAMES[predicted_index]
            confidence = float(prediction[0][predicted_index])
            predictions.append((predicted_label, confidence))

        if not predictions:
            return None

        return predictions

    def detect(self, landmarks, face_crop=None):
        if face_crop is not None:
            logger.info("Sleep detector received face crop shape %s", face_crop.shape)

        if not landmarks:
            logger.warning("No landmarks were provided for sleep detection.")
            return {
                "status": "Awake",
                "ear": 0.0,
                "source": "ear",
                "confidence": None,
            }

        left_eye = [landmarks[i] for i in self.LEFT_EYE if i < len(landmarks)]
        right_eye = [landmarks[i] for i in self.RIGHT_EYE if i < len(landmarks)]

        if len(left_eye) < 6 or len(right_eye) < 6:
            logger.warning("Insufficient landmarks for EAR-based sleep detection.")
            return {
                "status": "Awake",
                "ear": 0.0,
                "source": "ear",
                "confidence": None,
            }

        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2

        status = "Awake"
        source = "ear"
        confidence = None

        if ear < self.ear_threshold:
            status = "Sleeping"

        predictions = self._predict_with_model(face_crop, landmarks)
        if predictions:
            sleepy_count = sum(1 for label, _ in predictions if label == "sleepy")
            if sleepy_count > 0:
                status = "Sleeping"
            else:
                status = "Awake"
            source = "model"
            confidence = round(
                max(confidence or 0.0, max(score for _, score in predictions)), 3
            )

        logger.info(
            "Sleep detector result | EAR %.3f | Model confidence %s | Predicted status %s",
            round(ear, 3),
            confidence,
            status,
        )
        return {
            "status": status,
            "ear": round(ear, 3),
            "source": source,
            "confidence": confidence,
        }
