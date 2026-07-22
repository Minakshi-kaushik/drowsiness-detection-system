import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from inference.detect_image import ImageDetector
from inference.detect_video import VideoDetector

# Download YOLO model automatically on first run
if not os.path.exists("yolov8s.pt"):
    YOLO("yolov8s.pt")

SUPPORTED_IMAGE_TYPES = {"jpg", "jpeg", "png"}
SUPPORTED_VIDEO_TYPES = {"mp4", "avi", "mov"}


@st.cache_resource(show_spinner="Loading AI models...")
def get_image_detector() -> ImageDetector:
    return ImageDetector()


@st.cache_resource(show_spinner="Loading AI models...")
def get_video_detector() -> VideoDetector:
    return VideoDetector()


def save_uploaded_file(uploaded_file, suffix: str) -> Optional[str]:
    if uploaded_file is None:
        return None

    temp_dir = Path(tempfile.gettempdir()) / "drowsiness_streamlit"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uploaded_file.name}_{os.urandom(8).hex()}{suffix}"
    with temp_path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return str(temp_path)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    total_people = len(results or [])
    sleeping_people = sum(
        1
        for person in results or []
        if str(person.get("sleep_status") or "Unknown").lower()
        in {"sleeping", "sleepy"}
    )
    awake_people = total_people - sleeping_people
    return {
        "total_people": total_people,
        "sleeping_people": sleeping_people,
        "awake_people": awake_people,
    }


def format_confidence(confidence: Optional[float]) -> str:
    if confidence is None:
        return "N/A"
    return f"{float(confidence) * 100:.1f}%"


def render_person_cards(results: List[Dict[str, Any]]) -> None:
    if not results:
        st.info("No people detected.")
        return

    face_detected = any(
        person.get("face_detected") for person in results if isinstance(person, dict)
    )
    if not face_detected:
        st.info("No face detected.")
        return

    for index, person in enumerate(results, start=1):
        status = str(person.get("sleep_status") or "Unknown")
        is_sleeping = status.lower() in {"sleeping", "sleepy"}
        confidence = person.get("confidence")
        card_color = "#f8d7da" if is_sleeping else "#d4edda"
        border_color = "#dc3545" if is_sleeping else "#28a745"

        with st.container():
            st.markdown(
                f"<div style='border:2px solid {border_color}; border-radius:8px; padding:12px; background-color:{card_color}; margin-bottom:10px;'>"
                f"<b>Person #{index}</b><br>"
                f"<b>Status:</b> {status}<br>"
                f"<b>Confidence:</b> {format_confidence(confidence)}</div>",
                unsafe_allow_html=True,
            )


def display_image_result(annotated_image: np.ndarray) -> None:
    rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    st.image(rgb_image, caption="Annotated Image", use_container_width=True)


st.set_page_config(page_title="AI Drowsiness Detection System", layout="wide")

st.title("AI Drowsiness Detection System")

with st.sidebar:
    st.header("About")
    st.write(
        "Upload an image or video to detect people and identify whether they appear awake or sleeping."
    )

    st.header("Model Information")
    st.write("Backend: DetectionPipeline")
    st.write("Person detection: YOLOv8")
    st.write("Face detection: MediaPipe")
    st.write("Sleep detection: EAR-based classifier")

    st.header("Detection Statistics")
    st.write("Totals appear after a successful run.")

input_type = st.radio("Choose Input", ["Image", "Video"], horizontal=True)

if input_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=list(SUPPORTED_IMAGE_TYPES),
        help="Supported formats: jpg, jpeg, png",
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if Path(file_name).suffix.lstrip(".") not in SUPPORTED_IMAGE_TYPES:
            st.error("Unsupported file. Please upload a JPG, JPEG, or PNG image.")
        else:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            if st.button("Run Detection", use_container_width=True):
                try:
                    temp_path = save_uploaded_file(uploaded_file, ".png")
                    if not temp_path:
                        st.error("Processing failed.")
                        st.stop()

                    detector = get_image_detector()
                    annotated_image, results, output_path = detector.process(temp_path)
                    if annotated_image is None or annotated_image.size == 0:
                        st.error("Processing failed.")
                    else:
                        display_image_result(annotated_image)
                        summary = summarize_results(results)
                        st.subheader("Detection Summary")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total People", summary["total_people"])
                        col2.metric("Sleeping People", summary["sleeping_people"])
                        col3.metric("Awake People", summary["awake_people"])

                        st.subheader("Detected People")
                        render_person_cards(results)

                        if output_path and os.path.exists(output_path):
                            st.success(f"Processed image saved to: {output_path}")
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
    else:
        st.info("Upload an image to begin.")
else:
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=list(SUPPORTED_VIDEO_TYPES),
        help="Supported formats: mp4, avi, mov",
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if Path(file_name).suffix.lstrip(".") not in SUPPORTED_VIDEO_TYPES:
            st.error("Unsupported file. Please upload an MP4, AVI, or MOV video.")
        else:
            st.info("Processing video. This may take a moment.")
            if st.button("Run Detection", use_container_width=True):
                try:
                    temp_path = save_uploaded_file(uploaded_file, ".mp4")
                    if not temp_path:
                        st.error("Processing failed.")
                        st.stop()

                    detector = get_video_detector()
                    output_path, frame_results = detector.process(temp_path)
                    if not output_path or not os.path.exists(output_path):
                        st.error("Processing failed.")
                    else:
                        st.video(output_path)
                        summary = {
                            "total_people": 0,
                            "sleeping_people": 0,
                            "awake_people": 0,
                        }
                        if frame_results:
                            for frame_result in frame_results:
                                frame_summary = summarize_results(
                                    frame_result.get("results") or []
                                )
                                summary["total_people"] += frame_summary["total_people"]
                                summary["sleeping_people"] += frame_summary[
                                    "sleeping_people"
                                ]
                                summary["awake_people"] += frame_summary["awake_people"]
                        st.subheader("Detection Summary")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total People", summary["total_people"])
                        col2.metric("Sleeping People", summary["sleeping_people"])
                        col3.metric("Awake People", summary["awake_people"])
                        st.success(f"Processed video saved to: {output_path}")
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
    else:
        st.info("Upload a video to begin.")
