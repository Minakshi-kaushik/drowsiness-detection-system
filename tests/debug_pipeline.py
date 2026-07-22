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
    results = pipeline.process(image)

    print("People detected:", len(results))
    for person in results:
        print("Face detected:", person.get("face_detected"))
        print("Face crop size:", person.get("face_crop_size"))
        print("Age prediction:", person.get("age"))
        print(
            "Sleep prediction:",
            {
                "status": person.get("sleep_status"),
                "ear": person.get("ear"),
                "confidence": person.get("confidence"),
            },
        )
        print("Final dictionary:", person)
        print("-" * 60)


if __name__ == "__main__":
    main()
