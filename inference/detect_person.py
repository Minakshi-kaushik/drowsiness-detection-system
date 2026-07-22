"""
Person Detection Module
-----------------------
Detects all persons in an image using YOLOv8.

Author: Minakshi Kaushik
"""

from ultralytics import YOLO

from config.settings import YOLO_MODEL, PERSON_CONFIDENCE
from utils.logger import logger


class PersonDetector:
    """
    Detects all persons in an image using YOLOv8.
    """

    def __init__(self):
        logger.info("Loading YOLO Model...")

        self.model = YOLO(YOLO_MODEL)

        logger.info("YOLO Model Loaded Successfully.")

    def detect(self, image):
        """
        Detect all persons in an image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.

        Returns
        -------
        list
            List of detected people with metadata.
        """

        results = self.model(image, verbose=False)

        people = []

        image_height, image_width = image.shape[:2]

        person_id = 1

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])

                # Detect only persons (COCO class 0)
                if cls != 0:
                    continue

                if confidence < PERSON_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Keep bounding box inside image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_width, x2)
                y2 = min(image_height, y2)

                # Crop detected person
                crop = image[y1:y2, x1:x2].copy()

                people.append(
                    {
                        "id": person_id,
                        "bbox": (x1, y1, x2, y2),
                        "crop": crop,
                        "confidence": round(confidence, 3),
                        "width": x2 - x1,
                        "height": y2 - y1,
                    }
                )

                person_id += 1

        # Sort people from left to right
        people.sort(key=lambda person: person["bbox"][0])

        # Reassign IDs after sorting
        for index, person in enumerate(people, start=1):
            person["id"] = index

        logger.info(f"Detected {len(people)} person(s).")

        return people
