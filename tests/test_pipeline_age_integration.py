import unittest
from unittest.mock import patch

import numpy as np

from inference.pipeline import DetectionPipeline


class PipelineAgeIntegrationTests(unittest.TestCase):
    @patch("inference.detect_person.YOLO")
    @patch("inference.detect_face.mp.solutions.face_mesh.FaceMesh")
    @patch("inference.detect_sleep.tf.keras.models.load_model")
    def test_pipeline_returns_age_information_when_face_crop_available(
        self, mock_load_model, mock_face_mesh, mock_yolo
    ):
        mock_model = mock_yolo.return_value
        mock_result = type(
            "Result",
            (),
            {
                "boxes": [
                    type(
                        "Box",
                        (),
                        {
                            "cls": [np.array([0])],
                            "conf": [np.array([0.95])],
                            "xyxy": [np.array([0.0, 0.0, 100.0, 100.0])],
                        },
                    )()
                ]
            },
        )()
        mock_model.return_value = [mock_result]

        mock_face = type(
            "Face",
            (),
            {
                "landmark": [
                    type("LM", (), {"x": 0.2, "y": 0.2})(),
                    type("LM", (), {"x": 0.3, "y": 0.3})(),
                    type("LM", (), {"x": 0.4, "y": 0.4})(),
                    type("LM", (), {"x": 0.5, "y": 0.5})(),
                    type("LM", (), {"x": 0.6, "y": 0.6})(),
                    type("LM", (), {"x": 0.7, "y": 0.7})(),
                    type("LM", (), {"x": 0.8, "y": 0.8})(),
                    type("LM", (), {"x": 0.9, "y": 0.9})(),
                    type("LM", (), {"x": 0.1, "y": 0.1})(),
                    type("LM", (), {"x": 0.11, "y": 0.11})(),
                    type("LM", (), {"x": 0.12, "y": 0.12})(),
                    type("LM", (), {"x": 0.13, "y": 0.13})(),
                    type("LM", (), {"x": 0.14, "y": 0.14})(),
                    type("LM", (), {"x": 0.15, "y": 0.15})(),
                    type("LM", (), {"x": 0.16, "y": 0.16})(),
                    type("LM", (), {"x": 0.17, "y": 0.17})(),
                    type("LM", (), {"x": 0.18, "y": 0.18})(),
                    type("LM", (), {"x": 0.19, "y": 0.19})(),
                    type("LM", (), {"x": 0.20, "y": 0.20})(),
                    type("LM", (), {"x": 0.21, "y": 0.21})(),
                    type("LM", (), {"x": 0.22, "y": 0.22})(),
                ],
            },
        )()
        mock_mesh = mock_face_mesh.return_value
        mock_mesh.process.return_value = type(
            "Response",
            (),
            {"multi_face_landmarks": [mock_face]},
        )()

        mock_load_model.return_value = None

        pipeline = DetectionPipeline()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        results = pipeline.process(image)

        self.assertEqual(len(results), 1)
        self.assertIn("age", results[0])
        self.assertIn("available", results[0]["age"])


if __name__ == "__main__":
    unittest.main()
