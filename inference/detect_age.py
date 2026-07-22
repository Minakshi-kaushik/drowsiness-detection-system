"""
Age Prediction Module
---------------------
Predicts a person's age from a face crop using an OpenCV DNN AgeNet model.
The implementation is modular and fails gracefully if the pretrained model files
are not available in the repository.
"""

import os
import urllib.request

import cv2

# from jax import exc
import numpy as np

from config.settings import AGE_MODEL, AGE_PROTO, AGE_BUCKETS
from utils.logger import logger


def _read_net(proto_path, model_path):
    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        return None
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)


class AgeDetector:
    """Predicts an age group from a face crop."""

    def __init__(self, proto_path=None, model_path=None):
        self.proto_path = proto_path or AGE_PROTO
        self.model_path = model_path or AGE_MODEL
        self.model = self._load_model()

    def _ensure_model_files(self):
        missing_files = []
        if self.proto_path and not os.path.exists(self.proto_path):
            missing_files.append(self.proto_path)
        if self.model_path and not os.path.exists(self.model_path):
            missing_files.append(self.model_path)

        if not missing_files:
            return True

        logger.warning(
            "Age model not found. Missing file(s): %s", ", ".join(missing_files)
        )
        for missing_file in missing_files:
            directory = os.path.dirname(missing_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            if missing_file.endswith(".prototxt"):
                url = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt"
            elif missing_file.endswith(".caffemodel"):
                url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"
            else:
                continue

            try:
                logger.info(
                    "Attempting to download age model asset to %s from %s",
                    missing_file,
                    url,
                )
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
                with open(missing_file, "wb") as handle:
                    handle.write(payload)
                logger.info("Downloaded age model asset to %s", missing_file)
            except (
                Exception
            ) as exc:  # pragma: no cover - depends on network availability
                logger.exception(
                    "Unable to download age model asset for %s: %s", missing_file, exc
                )
                logger.warning(
                    "Place the missing file at %s before running age prediction.",
                    missing_file,
                )

        return os.path.exists(self.proto_path) and os.path.exists(self.model_path)

    def _load_model(self):
        print("\n========== LOADING AGE MODEL ==========")
        print("Proto Path :", self.proto_path)
        print("Model Path :", self.model_path)
        print("Proto Exists :", os.path.exists(self.proto_path))
        print("Model Exists :", os.path.exists(self.model_path))
        if not self.proto_path or not self.model_path:
            logger.warning("Age model paths are not configured.")
            return None

        if not self._ensure_model_files():
            logger.warning(
                "Age model files are still missing. Age prediction will be unavailable."
            )
            return None

        try:
            model = _read_net(self.proto_path, self.model_path)

            print("Model Object :", model)
            if model is None:
                logger.warning(
                    "OpenCV DNN AgeNet loader is unavailable in this environment. "
                    "Age prediction will be unavailable."
                )
                return None
            logger.info(
                "Loaded age model from %s and %s", self.proto_path, self.model_path
            )
            logger.info("Age model loaded successfully: %s", model is not None)
            return model
        except Exception as exc:
            print("\n===== OPENCV ERROR =====")
            print(type(exc))
            print(exc)
            print("========================\n")
            logger.exception("Unable to initialize age model: %s", exc)
            return None

    def _prepare_face(self, face_crop):
        if face_crop is None:
            return None

        if not isinstance(face_crop, np.ndarray):
            return None

        if face_crop.size == 0:
            return None

        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            return None

        if len(face_crop.shape) != 3 or face_crop.shape[2] < 3:
            logger.warning("Face crop is not a valid color image for age prediction.")
            return None

        height, width = face_crop.shape[:2]
        logger.info("Age detector received face crop shape %s", face_crop.shape)
        logger.info("Age detector expected model input shape (227, 227)")

        resized = cv2.resize(face_crop, (227, 227))
        blob = cv2.dnn.blobFromImage(
            resized,
            1.0,
            (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False,
        )
        return blob

    def predict(self, face_crop):
        """Predict age from a face crop.

        Returns a dictionary with the age prediction and confidence when available.
        """
        print("\n===== AGE DETECTOR =====")
        print("Model loaded :", self.model is not None)
        if self.model is None:
            logger.info("Age prediction unavailable because the model is not loaded.")
            return {"predicted_age": None, "confidence": None, "available": False}

        print("\nFace crop received by AgeDetector")
        print(type(face_crop))
        print(face_crop.shape)
        print(face_crop.dtype)
        print(face_crop.min(), face_crop.max())

        blob = self._prepare_face(face_crop)

        print("\nBlob:")
        print(blob.shape)
        print(blob.dtype)
        if blob is None:
            logger.warning("Received an invalid or empty face crop for age prediction.")
            return {"predicted_age": None, "confidence": None, "available": False}

        try:
            logger.info("Age prediction executed.")
            logger.info("Age detector input blob shape %s", blob.shape)

            self.model.setInput(blob)
            output = self.model.forward()

            print("\n========== AGE MODEL OUTPUT ==========")
            print(output)
            print("shape:", output.shape)
            print("======================================\n")

        except Exception as exc:
            print("\n========== AGE MODEL ERROR ==========")
            print(type(exc).__name__)
            print(exc)
            print("=====================================\n")

            logger.exception("Age prediction failed: %s", exc)

            return {
                "predicted_age": None,
                "confidence": None,
                "available": False,
            }
        if output is None or output.size == 0:
            logger.warning("Age model produced no output.")
            return {"predicted_age": None, "confidence": None, "available": False}

        index = int(np.argmax(output))
        confidence = float(output[0][index])
        age_range = AGE_BUCKETS[index] if index < len(AGE_BUCKETS) else str(index)
        result = {
            "predicted_age": age_range,
            "confidence": round(confidence, 3),
            "available": True,
        }
        logger.info("AgeDetector.predict returned: %s", result)
        print("Age prediction:", result)
        return result
