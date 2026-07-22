"""
Main Detection Service

This file will coordinate

Person Detection

↓

Face Detection

↓

Age Prediction

↓

Sleep Detection
"""

from inference.detect_person import PersonDetector


class DetectionService:
    def __init__(self):

        self.person_detector = PersonDetector()

    def detect(self, image):

        people = self.person_detector.detect(image)

        return people
