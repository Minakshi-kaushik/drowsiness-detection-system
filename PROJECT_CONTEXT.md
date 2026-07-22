# AI Drowsiness Detection System

## Objective

Build a professional AI-powered desktop application for detecting drowsiness in vehicle occupants.

The application must:

- Detect multiple people in an image or video.
- Count total people.
- Detect which people are sleeping.
- Draw green boxes for awake people.
- Draw red boxes for sleeping people.
- Predict the age of every detected person.
- Display popup messages summarizing detections.
- Support both image and video input.
- Save processed outputs.
- Have a professional Tkinter GUI.

---

## Technology Stack

Python 3.11

OpenCV

YOLOv8

MediaPipe Face Mesh

TensorFlow / Keras

Tkinter

NumPy

Matplotlib

---

## Current Project Structure

config/

datasets/

gui/

inference/

models/

outputs/

services/

tests/

training/

utils/

main.py

---

## Current Progress

Completed:

- Project architecture
- YOLOv8 person detector
- MediaPipe face detector
- TensorFlow CNN architecture
- Eye model training pipeline
- Eye model evaluation pipeline
- Eye prediction module
- Dataset preparation
- Unit tests for person and face detection

Not Completed:

- CNN integration into detect_sleep.py
- Master inference pipeline
- Age prediction
- GUI
- Video processing
- Deployment

---

## Coding Rules

- Keep code modular.
- Follow PEP8.
- Use logging instead of print().
- Keep all existing folder names.
- Do not change public APIs unless necessary.
- Write production-quality code.
- Keep Windows compatibility.
- Add docstrings.
- Avoid duplicate code.
- Prefer reusable classes.