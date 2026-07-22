"""
Face Detection Module
---------------------
Detects a face and facial landmarks inside a cropped person image
using MediaPipe Face Mesh.

Author: Minakshi Kaushik
"""

import cv2
import mediapipe as mp
import numpy as np

from utils.logger import logger


class FaceDetector:
    """
    Detects one face inside a person crop.
    """

    def __init__(self):

        logger.info("Loading MediaPipe Face Mesh...")

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        logger.info("MediaPipe Face Mesh Loaded.")

    def detect(self, person_crop):
        """
        Detect face inside one cropped person image.

        Parameters
        ----------
        person_crop : numpy.ndarray

        Returns
        -------
        dict | None
        """

        if person_crop is None:
            logger.warning("Face detection failed: person crop is None.")
            return None
        if not isinstance(person_crop, np.ndarray):
            logger.warning("Face detection failed: person crop is not a NumPy array.")
            return None
        if person_crop.size == 0:
            logger.warning("Face detection failed: person crop is empty.")
            return None
        if len(person_crop.shape) != 3 or person_crop.shape[2] < 3:
            logger.warning(
                "Face detection failed: person crop does not contain color channels."
            )
            return None
        if person_crop.shape[0] <= 0 or person_crop.shape[1] <= 0:
            logger.warning("Face detection failed: person crop has invalid dimensions.")
            return None
        if min(person_crop.shape[:2]) < 10:
            logger.warning(
                "Face detection failed: person crop is too small (%s).",
                person_crop.shape[:2],
            )
            return None

        height, width = person_crop.shape[:2]
        logger.info("Face detector received crop size %s x %s", width, height)

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            logger.warning(
                "Face detection failed: MediaPipe returned no landmarks for crop size %s.",
                person_crop.shape[:2],
            )
            return None

        face = results.multi_face_landmarks[0]

        h, w = person_crop.shape[:2]

        landmarks = []

        xs = []
        ys = []

        for landmark in face.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            landmarks.append((x, y))

            xs.append(x)
            ys.append(y)

        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(w, max(xs))
        y2 = min(h, max(ys))

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Face detection failed: invalid coordinates generated from landmarks: %s",
                (x1, y1, x2, y2),
            )
            return None

        logger.info("Face detector produced bbox %s", (x1, y1, x2, y2))
        return {"face_found": True, "bbox": (x1, y1, x2, y2), "landmarks": landmarks}
