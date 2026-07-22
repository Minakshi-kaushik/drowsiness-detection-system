import os
import sys

import cv2

sys.path.insert(0, os.getcwd())

from inference.pipeline import DetectionPipeline


def main():
    image_path = "assets/sample_images/driver/image_d1.jpeg"
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"Unable to load image: {image_path}")

    pipeline = DetectionPipeline()
    persons = pipeline.person_detector.detect(image) if pipeline.person_detector else []
    results = pipeline.process(image)

    print("Detected people:", len(persons))
    detected_faces = sum(1 for person in results if person.get("face_detected"))
    print("Detected faces:", detected_faces)

    for person in persons:
        person_id = person.get("id")
        crop = person.get("crop")
        face = None
        if pipeline.face_detector is not None:
            face = pipeline.face_detector.detect(crop)

        face_crop = None
        if face is not None and face.get("bbox") is not None:
            x1, y1, x2, y2 = face["bbox"]
            face_crop = crop[y1:y2, x1:x2].copy()

        print("Person ID:", person_id)
        print("Bounding box:", person.get("bbox"))
        print("Face found:", face is not None)
        print("Face crop shape:", None if face_crop is None else face_crop.shape)

        age_result = None
        if pipeline.age_detector is not None and face_crop is not None:
            age_result = pipeline.age_detector.predict(face_crop)

        print("Input image shape:", None if face_crop is None else face_crop.shape)
        print("Expected model input shape:", (227, 227))
        print("Model loaded successfully:", pipeline.age_detector.model is not None)
        print("Prediction result:", age_result)

        sleep_result = None
        if pipeline.sleep_detector is not None and face_crop is not None:
            sleep_result = pipeline.sleep_detector.detect(
                face.get("landmarks"), face_crop=face_crop
            )

        print("EAR:", None if sleep_result is None else sleep_result.get("ear"))
        print(
            "Model confidence:",
            None if sleep_result is None else sleep_result.get("confidence"),
        )
        print(
            "Predicted status:",
            None if sleep_result is None else sleep_result.get("status"),
        )

        result_entry = next(
            (item for item in results if item.get("id") == person_id), {}
        )
        print("Final dictionary:", result_entry)
        print("-" * 60)


if __name__ == "__main__":
    main()
