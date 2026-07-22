"""Video inference entry point for running the backend detection pipeline on a video."""

import os

import cv2

from config.settings import OUTPUT_VIDEO_DIR
from inference.pipeline import DetectionPipeline
from utils.drawing import DrawingUtils
from utils.logger import logger


class VideoDetector:
    """Run the full detection pipeline on each frame of a video file."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline or DetectionPipeline()

    def _validate_video_path(self, video_path):
        if not video_path:
            raise ValueError("A video path is required.")
        if not os.path.isfile(video_path):
            raise ValueError(f"Video path does not exist: {video_path}")
        if os.path.getsize(video_path) <= 0:
            raise ValueError("Video file is empty.")

    def process(self, video_path):
        """Process each frame of the video and save an annotated copy."""
        self._validate_video_path(video_path)

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("Unable to open video.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        output_name = (
            f"{os.path.splitext(os.path.basename(video_path))[0]}_annotated.mp4"
        )
        output_path = os.path.join(OUTPUT_VIDEO_DIR, output_name)
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if output_name.lower().endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        writer = cv2.VideoWriter(
            output_path, fourcc, fps if fps > 0 else 24.0, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError("Unable to initialize video writer.")

        frame_results = []
        frame_index = 0
        try:
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    break
                if frame.size == 0:
                    logger.warning("Skipping an empty frame from %s", video_path)
                    continue

                annotated_frame = frame.copy()
                results = self.pipeline.process(annotated_frame) or []
                logger.info(
                    "Video frame %s pipeline results: %s", frame_index, results
                )
                print(f"Video frame {frame_index} results:", results)

                for person in results:
                    DrawingUtils.draw_person_box(annotated_frame, person)

                awake_count = sum(
                    1
                    for person in results
                    if str(person.get("sleep_status") or "Unknown").lower()
                    not in {"sleeping", "sleepy"}
                )
                sleeping_count = len(results) - awake_count
                DrawingUtils.draw_summary(
                    annotated_frame,
                    len(results),
                    awake_count,
                    sleeping_count,
                )

                writer.write(annotated_frame)
                frame_results.append({"frame": frame_index, "results": results})
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        logger.info("Video inference completed for %s", video_path)
        return output_path, frame_results
