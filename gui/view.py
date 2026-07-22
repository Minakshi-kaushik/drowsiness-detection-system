"""Tkinter view for the drowsiness detection application."""

import os
import tkinter as tk
from tkinter import ttk
from typing import Optional

from PIL import Image, ImageTk

from config.settings import OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR
from utils.drawing import normalize_person_result


class AppView(tk.Tk):
    """Main application window with left, center, right, and bottom regions."""

    def __init__(self, controller):
        super().__init__()
        self.title("Drowsiness Detection System")
        self.geometry("1200x760")
        self.minsize(1000, 700)
        self.controller = controller

        self._build_layout()
        self._bind_actions()

    def _build_layout(self):
        self.configure(bg="#f2f4f8")

        self.left_panel = ttk.Frame(self, padding=12)
        self.left_panel.grid(row=0, column=0, sticky="nswe", padx=(12, 6), pady=12)

        self.center_panel = ttk.LabelFrame(self, text="Preview", padding=12)
        self.center_panel.grid(row=0, column=1, sticky="nswe", padx=6, pady=12)

        self.right_panel = ttk.LabelFrame(self, text="Detection Summary", padding=12)
        self.right_panel.grid(row=0, column=2, sticky="nswe", padx=(6, 12), pady=12)

        self.bottom_bar = ttk.Frame(self, padding=(12, 6, 12, 12))
        self.bottom_bar.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.left_panel.columnconfigure(0, weight=1)
        self.center_panel.columnconfigure(0, weight=1)
        self.right_panel.columnconfigure(0, weight=1)

        self._build_left_controls()
        self._build_preview_area()
        self._build_summary_area()
        self._build_status_bar()

    def _build_left_controls(self):
        title = ttk.Label(
            self.left_panel, text="Drowsiness Detection", font=("Segoe UI", 18, "bold")
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 20))

        ttk.Button(
            self.left_panel, text="Upload Image", command=self.controller.upload_image
        ).grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(
            self.left_panel, text="Upload Video", command=self.controller.upload_video
        ).grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(
            self.left_panel,
            text="Start Detection",
            command=self.controller.start_detection,
        ).grid(row=3, column=0, sticky="ew", pady=4)
        ttk.Button(
            self.left_panel, text="Save Output", command=self.controller.save_output
        ).grid(row=4, column=0, sticky="ew", pady=4)
        ttk.Button(self.left_panel, text="Clear", command=self.controller.clear).grid(
            row=5, column=0, sticky="ew", pady=4
        )

    def _build_preview_area(self):
        self.preview_label = ttk.Label(
            self.center_panel, text="No media selected", anchor="center"
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_image = None
        self.preview_photo = None

    def _build_summary_area(self):
        self.summary_labels = {}
        for index, label in enumerate(
            [
                "Total People",
                "Sleeping People",
                "Awake People",
                "Processing Time",
            ]
        ):
            ttk.Label(self.right_panel, text=label, font=("Segoe UI", 10, "bold")).grid(
                row=index * 2, column=0, sticky="w", pady=(8, 2)
            )
            value_label = ttk.Label(self.right_panel, text="-", foreground="#2f4f4f")
            value_label.grid(row=index * 2 + 1, column=0, sticky="w", pady=(0, 10))
            self.summary_labels[label] = value_label

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.bottom_bar, textvariable=self.status_var).pack(side="left")

    def _bind_actions(self):
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.destroy()

    def display_image_preview(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
            image.thumbnail((700, 550))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_photo)
            self.preview_label.image = self.preview_photo
        except Exception:
            self.preview_label.configure(text="Unable to preview image")

    def display_video_preview(self, video_path):
        self.preview_label.configure(
            text=f"Video selected: {os.path.basename(video_path)}"
        )

    def show_processed_image(self, image):
        try:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.convert("RGB")
            pil_image.thumbnail((700, 550))
            self.preview_photo = ImageTk.PhotoImage(pil_image)
            self.preview_label.configure(image=self.preview_photo)
            self.preview_label.image = self.preview_photo
        except Exception:
            self.preview_label.configure(text="Unable to display processed image")

    def show_processed_video(self, output_path):
        self.preview_label.configure(text=f"Processed video saved to:\n{output_path}")

    def clear_preview(self):
        self.preview_label.configure(text="No media selected", image="")
        self.preview_label.image = None
        self.preview_photo = None

    def set_status(self, message):
        self.status_var.set(message)

    def set_processing_time(self, seconds):
        self.summary_labels["Processing Time"].configure(text=f"{seconds:.2f}s")

    def update_summary(self, results):
        normalized_results = [
            normalize_person_result(person) for person in results or []
        ]
        total_people = len(normalized_results)
        sleeping_people = sum(
            1
            for person in normalized_results
            if str(person.get("sleep_status") or "Unknown").lower()
            in {"sleeping", "sleepy"}
        )
        awake_people = total_people - sleeping_people
        self.summary_labels["Total People"].configure(text=str(total_people))
        self.summary_labels["Sleeping People"].configure(text=str(sleeping_people))
        self.summary_labels["Awake People"].configure(text=str(awake_people))

    def update_summary_from_frames(self, frame_results):
        total_people = 0
        sleeping_people = 0
        for frame in frame_results or []:
            for person in frame.get("results") or []:
                normalized_person = normalize_person_result(person)
                total_people += 1
                if str(normalized_person.get("sleep_status") or "Unknown").lower() in {
                    "sleeping",
                    "sleepy",
                }:
                    sleeping_people += 1
        awake_people = total_people - sleeping_people
        self.summary_labels["Total People"].configure(text=str(total_people))
        self.summary_labels["Sleeping People"].configure(text=str(sleeping_people))
        self.summary_labels["Awake People"].configure(text=str(awake_people))

    def reset_summary(self):
        self.summary_labels["Total People"].configure(text="-")
        self.summary_labels["Sleeping People"].configure(text="-")
        self.summary_labels["Awake People"].configure(text="-")
        self.summary_labels["Processing Time"].configure(text="-")
