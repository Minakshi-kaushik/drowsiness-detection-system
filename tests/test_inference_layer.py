import os

import cv2
import numpy as np
import pytest

from inference.detect_image import ImageDetector
from inference.detect_video import VideoDetector


class DummyPipeline:
    def __init__(self):
        self.calls = 0

    def process(self, image):
        self.calls += 1
        return [
            {
                "id": 1,
                "person_bbox": (0, 0, 50, 50),
                "person_confidence": 0.95,
                "face_detected": True,
                "face_bbox": (5, 5, 20, 20),
                "sleep_status": "Awake",
                "ear": 0.31,
                "age": {
                    "predicted_age": "(25-32)",
                    "confidence": 0.88,
                    "available": True,
                },
            }
        ]


def test_image_detector_processes_valid_image(tmp_path):
    image_path = tmp_path / "valid.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))

    detector = ImageDetector(pipeline=DummyPipeline())
    annotated_image, results, output_path = detector.process(str(image_path))

    assert annotated_image is not None
    assert len(results) == 1
    assert results[0]["id"] == 1
    assert os.path.exists(output_path)


def test_image_detector_rejects_invalid_image_path(tmp_path):
    detector = ImageDetector(pipeline=DummyPipeline())

    with pytest.raises(ValueError):
        detector.process(str(tmp_path / "missing.jpg"))


def test_image_detector_rejects_empty_image(tmp_path):
    image_path = tmp_path / "empty.jpg"
    image_path.write_bytes(b"")

    detector = ImageDetector(pipeline=DummyPipeline())

    with pytest.raises(ValueError):
        detector.process(str(image_path))


def test_video_detector_processes_valid_video(tmp_path):
    video_path = tmp_path / "valid_video.avi"

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64)
    )
    if not writer.isOpened():
        pytest.skip("VideoWriter is unavailable in this environment")

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    writer.write(frame)
    writer.write(frame)
    writer.release()

    detector = VideoDetector(pipeline=DummyPipeline())
    output_path, frame_results = detector.process(str(video_path))

    assert os.path.exists(output_path)
    assert len(frame_results) == 2


def test_video_detector_rejects_invalid_video_path(tmp_path):
    detector = VideoDetector(pipeline=DummyPipeline())

    with pytest.raises(ValueError):
        detector.process(str(tmp_path / "missing.mp4"))


def test_video_detector_skips_empty_frame(monkeypatch, tmp_path):
    class FakeCapture:
        def __init__(self):
            self.read_calls = 0

        def isOpened(self):
            return True

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FPS:
                return 10.0
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 64
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 64
            return 0

        def read(self):
            self.read_calls += 1
            if self.read_calls == 1:
                return True, None
            return False, None

        def release(self):
            return None

    class FakeWriter:
        def __init__(self, *args, **kwargs):
            self.written = 0

        def isOpened(self):
            return True

        def write(self, frame):
            self.written += 1

        def release(self):
            return None

    video_path = tmp_path / "empty_frame.avi"
    video_path.write_bytes(b"fake")

    monkeypatch.setattr(
        "inference.detect_video.cv2.VideoCapture", lambda *_: FakeCapture()
    )
    monkeypatch.setattr(
        "inference.detect_video.cv2.VideoWriter", lambda *args, **kwargs: FakeWriter()
    )

    detector = VideoDetector(pipeline=DummyPipeline())
    output_path, frame_results = detector.process(str(video_path))

    assert output_path is not None
    assert frame_results == []
