"""Image inference entry point for running the backend detection pipeline."""

import os

import cv2

from config.settings import OUTPUT_IMAGE_DIR
from inference.pipeline import DetectionPipeline
from utils.drawing import DrawingUtils
from utils.logger import logger


class ImageDetector:
    """Run the full detection pipeline on a single image file."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline or DetectionPipeline()

    def _validate_image_path(self, image_path):
        if not image_path:
            raise ValueError("An image path is required.")
        if not os.path.isfile(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")
        if os.path.getsize(image_path) <= 0:
            raise ValueError("Image file is empty.")

    def process(self, image_path):
        """Load an image, run the detection pipeline, and save an annotated copy."""
        self._validate_image_path(image_path)

        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            raise ValueError("Unable to load image.")

        annotated_image = image.copy()
        results = self.pipeline.process(image)
        logger.info("Pipeline returned %s result(s)", len(results or []))
        logger.info("ImageDetector.process pipeline results: %s", results or [])

        for person in results or []:
            DrawingUtils.draw_person_box(annotated_image, person)
            logger.info("Drawing executed for Person %s", person.get("id"))

        awake_count = sum(
            1
            for person in results or []
            if str(person.get("sleep_status") or "Unknown").lower()
            not in {"sleeping", "sleepy"}
        )
        sleeping_count = len(results or []) - awake_count

        DrawingUtils.draw_summary(
            annotated_image, len(results or []), awake_count, sleeping_count
        )

        output_name = (
            f"{os.path.splitext(os.path.basename(image_path))[0]}_annotated.jpg"
        )
        output_path = DrawingUtils.save_output(
            annotated_image, OUTPUT_IMAGE_DIR, output_name
        )

        logger.info("Returned annotated image: %s", output_path)
        logger.info("Image inference completed for %s", image_path)
        return annotated_image, results or [], output_path
