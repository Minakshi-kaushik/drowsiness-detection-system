from utils.drawing import normalize_person_result


def test_normalize_person_result_exposes_gui_values_without_losing_fields():
    person = {
        "id": 1,
        "person_bbox": (0, 0, 10, 10),
        "person_confidence": 0.91,
        "face_detected": True,
        "sleep_status": "Awake",
        "confidence": 0.91,
        "age": {"predicted_age": "(25-32)", "confidence": 0.87, "available": True},
    }

    normalized = normalize_person_result(person)

    assert normalized["id"] == 1
    assert normalized["sleep_status"] == "Awake"
    assert normalized["status"] == "Awake"
    assert normalized["confidence"] == 0.91
    assert normalized["age"]["predicted_age"] == "(25-32)"
    assert "age_value" not in normalized
