"""Controller layer for the Tkinter GUI."""

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

from config.settings import OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR
from inference.detect_image import ImageDetector
from inference.detect_video import VideoDetector
from utils.logger import logger


class AppController:
    """Coordinate user actions between the Tkinter view and backend detectors."""

    def __init__(self, view):
        self.view = view
        self.image_detector = ImageDetector()
        self.video_detector = VideoDetector()
        self.current_image_path: Optional[str] = None
        self.current_video_path: Optional[str] = None
        self.current_output_path: Optional[str] = None
        self._processing_thread: Optional[threading.Thread] = None

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if path:
            self.current_image_path = path
            self.view.set_status(f"Loaded image: {os.path.basename(path)}")
            self.view.display_image_preview(path)

    def upload_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")],
        )
        if path:
            self.current_video_path = path
            self.view.set_status(f"Loaded video: {os.path.basename(path)}")
            self.view.display_video_preview(path)

    def start_detection(self):
        if self.current_image_path:
            self._run_image_detection(self.current_image_path)
        elif self.current_video_path:
            self._run_video_detection(self.current_video_path)
        else:
            messagebox.showwarning("No input", "Please upload an image or video first.")

    def clear(self):
        self.current_image_path = None
        self.current_video_path = None
        self.current_output_path = None
        self.view.clear_preview()
        self.view.reset_summary()
        self.view.set_status("Ready")

    def save_output(self):
        if not self.current_output_path:
            messagebox.showinfo("No output", "No processed output is available yet.")
            return

        default_dir = OUTPUT_IMAGE_DIR if self.current_image_path else OUTPUT_VIDEO_DIR
        target_path = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=os.path.basename(self.current_output_path),
            title="Save processed output",
        )
        if target_path:
            try:
                import shutil

                shutil.copyfile(self.current_output_path, target_path)
                self.view.set_status(f"Saved to {os.path.basename(target_path)}")
            except Exception as exc:  # pragma: no cover - GUI error path
                logger.exception("Unable to save output: %s", exc)
                messagebox.showerror("Save failed", str(exc))

    def _dispatch_to_main_thread(self, callback, *args, **kwargs):
        if self.view is None:
            return
        self.view.after(0, lambda: callback(*args, **kwargs))

    def _run_image_detection(self, image_path):
        self._processing_thread = threading.Thread(
            target=self._process_image_thread,
            args=(image_path,),
            daemon=True,
        )
        self._processing_thread.start()

    def _run_video_detection(self, video_path):
        self._processing_thread = threading.Thread(
            target=self._process_video_thread,
            args=(video_path,),
            daemon=True,
        )
        self._processing_thread.start()

    def _process_image_thread(self, image_path):
        self._dispatch_to_main_thread(self.view.set_status, "Processing...")
        start_time = time.time()
        try:
            annotated_image, results, output_path = self.image_detector.process(
                image_path
            )
            logger.info("GUI payload results: %s", results)
            print("GUI payload:", results)
            self.current_output_path = output_path
            self._dispatch_to_main_thread(
                self.view.show_processed_image, annotated_image
            )
            self._dispatch_to_main_thread(self.view.update_summary, results)
            self._dispatch_to_main_thread(self.view.set_status, "Completed")
            self._dispatch_to_main_thread(
                self.view.set_processing_time, time.time() - start_time
            )
        except Exception as exc:  # pragma: no cover - GUI error path
            logger.exception("Image processing failed: %s", exc)
            self._dispatch_to_main_thread(self.view.set_status, "Error")
            self._dispatch_to_main_thread(
                messagebox.showerror, "Processing failed", str(exc)
            )

    def _process_video_thread(self, video_path):
        self._dispatch_to_main_thread(self.view.set_status, "Processing...")
        start_time = time.time()
        try:
            output_path, frame_results = self.video_detector.process(video_path)
            self.current_output_path = output_path
            self._dispatch_to_main_thread(self.view.show_processed_video, output_path)
            self._dispatch_to_main_thread(
                self.view.update_summary_from_frames, frame_results
            )
            self._dispatch_to_main_thread(self.view.set_status, "Completed")
            self._dispatch_to_main_thread(
                self.view.set_processing_time, time.time() - start_time
            )
        except Exception as exc:  # pragma: no cover - GUI error path
            logger.exception("Video processing failed: %s", exc)
            self._dispatch_to_main_thread(self.view.set_status, "Error")
            self._dispatch_to_main_thread(
                messagebox.showerror, "Processing failed", str(exc)
            )
