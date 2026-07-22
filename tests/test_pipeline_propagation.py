import numpy as np

from inference.pipeline import DetectionPipeline


class FakePersonDetector:
    def detect(self, image):
        return [
            {
                "id": 1,
                "bbox": (0, 0, 10, 10),
                "crop": np.zeros((10, 10, 3), dtype=np.uint8),
                "confidence": 0.95,
            }
        ]


class FakeFaceDetector:
    def detect(self, crop):
        return {"bbox": (0, 0, 5, 5), "landmarks": [(0, 0)] * 10}


class FakeSleepDetector:
    def detect(self, landmarks, face_crop=None):
        return {"status": "Awake", "ear": 0.25, "confidence": 0.91}


class FakeAgeDetector:
    def predict(self, face_crop):
        return {"predicted_age": "(25-32)", "confidence": 0.88, "available": True}


def test_pipeline_propagates_age_and_sleep_fields():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline.person_detector = FakePersonDetector()
    pipeline.face_detector = FakeFaceDetector()
    pipeline.sleep_detector = FakeSleepDetector()
    pipeline.age_detector = FakeAgeDetector()

    results = pipeline.process(np.zeros((20, 20, 3), dtype=np.uint8))

    assert len(results) == 1
    person = results[0]
    assert person["age"]["predicted_age"] == "(25-32)"
    assert person["status"] == "Awake"
    assert person["confidence"] == 0.91
