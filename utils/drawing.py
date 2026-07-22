"""Utilities for drawing detection overlays and saving annotated outputs."""

import os

import cv2

from config.settings import (
    COLOR_AWAKE,
    COLOR_SLEEP,
    COLOR_TEXT,
    FONT_SCALE,
    FONT_THICKNESS,
)
from utils.helpers import create_directory
from utils.logger import logger


def normalize_person_result(person_result):
    """Return a GUI-safe person dictionary while preserving all pipeline fields."""
    if not person_result:
        return person_result

    normalized = dict(person_result)
    if "status" not in normalized:
        normalized["status"] = normalized.get("sleep_status") or "Unknown"
    if "sleep_status" not in normalized:
        normalized["sleep_status"] = normalized.get("status") or "Unknown"

    if "confidence" not in normalized:
        normalized["confidence"] = normalized.get("person_confidence")

    if normalized.get("confidence") is None:
        normalized["confidence"] = normalized.get("person_confidence")

    return normalized


class DrawingUtils:
    """Centralized helpers for drawing inference overlays."""

    @staticmethod
    def draw_person_box(image, person_result):
        """Draw a bounding box and metadata for one detected person."""
        if image is None or image.size == 0 or not person_result:
            return image

        normalized_person = normalize_person_result(person_result)
        bbox = normalized_person.get("person_bbox") or normalized_person.get("bbox")
        if bbox is None:
            return image

        x1, y1, x2, y2 = [int(value) for value in bbox]
        status = str(normalized_person.get("sleep_status") or "Unknown")
        color = COLOR_SLEEP if status.lower() in {"sleeping", "sleepy"} else COLOR_AWAKE

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        DrawingUtils.draw_labels(image, normalized_person, (x1, y1), color)
        logger.info("Drawing executed for Person %s", normalized_person.get("id"))
        return image

    @staticmethod
    def draw_labels(image, person_result, position, color):
        """Draw a compact label with person ID, sleep state, age, and confidence."""
        if image is None or image.size == 0 or not person_result:
            return image

        normalized_person = normalize_person_result(person_result)
        x1, y1 = position
        person_id = normalized_person.get("id", "?")
        sleep_status = str(normalized_person.get("sleep_status") or "Unknown")

        confidence_value = normalized_person.get("person_confidence")
        if confidence_value is None:
            confidence_value = normalized_person.get("confidence")

        confidence_text = ""
        if confidence_value is not None:
            confidence_text = f" | Conf {float(confidence_value):.2f}"

        label = f"Person {person_id} | {sleep_status}{confidence_text}"
        logger.info("Final overlay text: %s", label)
        cv2.putText(
            image,
            label,
            (max(0, x1), max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            COLOR_TEXT,
            FONT_THICKNESS,
        )
        return image

    @staticmethod
    def draw_people(image, people):
        """Backward-compatible wrapper that draws all detected people."""
        for person in people or []:
            DrawingUtils.draw_person_box(image, person)
        return image

    @staticmethod
    def draw_summary(image, total_people, awake_people, sleeping_people):
        """Draw a summary block with totals for the frame."""
        if image is None or image.size == 0:
            return image

        text = f"Total People: {total_people} | Awake: {awake_people} | Sleeping: {sleeping_people}"
        cv2.putText(
            image,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_TEXT,
            2,
        )
        logger.info("Drawing summary executed: %s", text)
        return image

    @staticmethod
    def draw_statistics(image, people):
        """Backward-compatible wrapper for a summary based on detected people."""
        awake_count = 0
        sleeping_count = 0
        for person in people or []:
            status = str(person.get("sleep_status") or "Unknown")
            if status.lower() in {"sleeping", "sleepy"}:
                sleeping_count += 1
            else:
                awake_count += 1

        return DrawingUtils.draw_summary(
            image, len(people or []), awake_count, sleeping_count
        )

    @staticmethod
    def save_output(image, output_dir, filename):
        """Create the output directory, save the annotated image/video frame, and return the path."""
        if image is None or image.size == 0:
            raise ValueError("Cannot save an empty image.")

        create_directory(output_dir)
        output_path = os.path.join(output_dir, filename)
        success = cv2.imwrite(output_path, image)
        if not success:
            raise IOError(f"Unable to save output image to {output_path}")

        logger.info("Saved annotated output to %s", output_path)
        return output_path
