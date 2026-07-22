import cv2
import numpy as np

from utils.drawing import DrawingUtils, normalize_person_result


def test_normalize_person_result_does_not_expose_age_value(monkeypatch):
    person = {
        "id": 1,
        "sleep_status": "Awake",
        "person_confidence": 0.91,
        "age": {"predicted_age": "(25-32)", "confidence": 0.87, "available": True},
    }

    normalized = normalize_person_result(person)

    assert "age_value" not in normalized
    assert "age" in normalized


def test_draw_labels_omits_age_information(monkeypatch):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    captured = {}

    def fake_put_text(_image, text, _position, _font, _scale, _color, _thickness):
        captured["text"] = text

    monkeypatch.setattr(cv2, "putText", fake_put_text)

    DrawingUtils.draw_labels(
        image,
        {
            "id": 2,
            "sleep_status": "Sleeping",
            "age": {"predicted_age": "(25-32)"},
            "confidence": 0.88,
        },
        (10, 50),
        (0, 255, 0),
    )

    text = captured["text"]
    assert "Age" not in text
    assert "Person 2" in text
    assert "Sleeping" in text
    assert "Conf 0.88" in text
