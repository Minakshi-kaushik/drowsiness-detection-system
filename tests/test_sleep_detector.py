import cv2

from inference.detect_face import FaceDetector
from inference.detect_sleep import SleepDetector

image = cv2.imread("assets/sample_images/single/image1.jpeg")

face_detector = FaceDetector()

sleep_detector = SleepDetector()

faces = face_detector.detect(image)

print("=" * 40)

for face in faces:
    result = sleep_detector.detect(face["landmarks"])

    print("Face:", face["id"])
    print("Status:", result["status"])
    print("EAR:", result["ear"])
    print()
