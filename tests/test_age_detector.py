import unittest
from unittest.mock import patch

import numpy as np

from inference.detect_age import AgeDetector


class AgeDetectorTests(unittest.TestCase):
    def test_predict_returns_unavailable_when_model_missing(self):
        detector = AgeDetector(
            proto_path="missing.prototxt", model_path="missing.caffemodel"
        )
        result = detector.predict(np.zeros((50, 50, 3), dtype=np.uint8))

        self.assertFalse(result["available"])
        self.assertIsNone(result["predicted_age"])
        self.assertIsNone(result["confidence"])

    def test_predict_returns_unavailable_for_empty_face_crop(self):
        detector = AgeDetector(
            proto_path="missing.prototxt", model_path="missing.caffemodel"
        )
        result = detector.predict(np.empty((0, 0, 3), dtype=np.uint8))

        self.assertFalse(result["available"])
        self.assertIsNone(result["predicted_age"])
        self.assertIsNone(result["confidence"])

    def test_predict_uses_mocked_model(self):
        detector = AgeDetector(proto_path="fake.prototxt", model_path="fake.caffemodel")
        detector.model = type(
            "MockModel",
            (),
            {"forward": lambda self: np.array([[0.0, 0.9, 0.1]])},
        )()
        detector.model.setInput = lambda blob: None

        result = detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertTrue(result["available"])
        self.assertEqual(result["predicted_age"], "(4-6)")
        self.assertAlmostEqual(result["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
