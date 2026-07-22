"""
Detection Pipeline
------------------
Runs the complete AI pipeline.

Flow:
Image
  ↓
YOLO Person Detection
  ↓
Crop Each Person
  ↓
Face Detection
  ↓
Sleep Detection
  ↓
Age Prediction (Coming Soon)

Author: Minakshi Kaushik
"""

from utils.logger import logger


class DetectionPipeline:
    def __init__(self):

        logger.info("Initializing Detection Pipeline...")

        self.person_detector = None
        self.face_detector = None
        self.sleep_detector = None
        self.age_detector = None

        self._load_components()

    def _load_components(self):
        try:
            from inference.detect_age import AgeDetector
            from inference.detect_face import FaceDetector
            from inference.detect_person import PersonDetector
            from inference.detect_sleep import SleepDetector
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            logger.warning("Unable to load detection components: %s", exc)
            return

        try:
            self.person_detector = PersonDetector()
            logger.info("Person detector initialized.")
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            logger.warning("Person detector initialization failed: %s", exc)

        try:
            self.face_detector = FaceDetector()
            logger.info("Face detector initialized.")
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            logger.warning("Face detector initialization failed: %s", exc)

        try:
            self.sleep_detector = SleepDetector()
            logger.info("Sleep detector initialized.")
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            logger.warning("Sleep detector initialization failed: %s", exc)

        try:
            self.age_detector = AgeDetector()
            logger.info("Age detector initialized.")
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            logger.warning("Age detector initialization failed: %s", exc)

        logger.info("Detection Pipeline Ready.")

    def process(self, image):

        if image is None or image.size == 0:
            logger.warning("Received an empty image for pipeline processing.")
            return []

        if self.person_detector is None:
            logger.warning("Person detector is unavailable.")
            return []

        final_results = []

        persons = self.person_detector.detect(image)

        logger.info("Image loaded for pipeline processing.")
        logger.info("People detected: %s", len(persons))

        for person in persons:
            crop = person.get("crop")
            logger.info(
                "Processing person %s with crop size %s",
                person.get("id"),
                None if crop is None else crop.shape[:2],
            )

            face = None
            if self.face_detector is not None:
                face = self.face_detector.detect(crop)
                face_bbox = face.get("bbox") if face else None
                logger.info(
                    "Person ID %s | Bounding box %s | Face found %s",
                    person.get("id"),
                    face_bbox,
                    bool(face and face.get("face_found", False)),
                )

            if face is None:
                logger.warning("No face detected for Person %s", person.get("id"))

                final_results.append(
                    {
                        "id": person["id"],
                        "person_bbox": person["bbox"],
                        "person_confidence": person["confidence"],
                        "face_detected": False,
                        "face_bbox": None,
                        "landmarks": None,
                        "face_crop_size": None,
                        "sleep_status": "Unknown",
                        "status": "Unknown",
                        "ear": None,
                        "confidence": person.get("confidence"),
                        "age": None,
                    }
                )

                continue

            face_bbox = face.get("bbox")
            face_crop = None

            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                face_crop = crop[y1:y2, x1:x2].copy()

            if face_crop is not None:
                logger.info(
                    "Person ID %s | Face crop size %s x %s",
                    person.get("id"),
                    face_crop.shape[1],
                    face_crop.shape[0],
                )
                logger.info(
                    "Person ID %s | Face crop valid for age prediction: %s",
                    person.get("id"),
                    face_crop.size > 0 and len(face_crop.shape) == 3,
                )

            sleep_result = {"status": "Unknown", "ear": None, "confidence": None}
            if self.sleep_detector is not None:
                sleep_result = self.sleep_detector.detect(
                    face.get("landmarks"), face_crop=face_crop
                )
                logger.info(
                    "Person ID %s | Sleep prediction executed: %s",
                    person.get("id"),
                    sleep_result,
                )

            age_result = None
            if self.age_detector is not None:
                age_result = self.age_detector.predict(face_crop)
                logger.info(
                    "Person ID %s | Age prediction executed: %s",
                    person.get("id"),
                    age_result,
                )

            sleep_status = sleep_result.get("status", "Unknown")
            sleep_confidence = sleep_result.get("confidence")
            final_results.append(
                {
                    "id": person["id"],
                    "person_bbox": person["bbox"],
                    "person_confidence": person["confidence"],
                    "face_detected": True,
                    "face_bbox": face["bbox"],
                    "landmarks": face["landmarks"],
                    "face_crop_size": None
                    if face_crop is None
                    else (face_crop.shape[1], face_crop.shape[0]),
                    "sleep_status": sleep_status,
                    "status": sleep_status,
                    "ear": sleep_result.get("ear"),
                    "confidence": sleep_confidence
                    if sleep_confidence is not None
                    else person.get("confidence"),
                    "age": age_result,
                }
            )

        logger.info(f"Pipeline finished with {len(final_results)} result(s).")
        logger.info("Pipeline final_results: %s", final_results)
        print("Pipeline output:", final_results)

        return final_results
