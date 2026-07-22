import cv2

from inference.detect_person import PersonDetector
from inference.detect_face import FaceDetector

image = cv2.imread("assets/sample_images/single/image1.jpeg")

person_detector = PersonDetector()

face_detector = FaceDetector()

people = person_detector.detect(image)

print("=" * 60)

print(f"People Detected : {len(people)}")

print("=" * 60)

for person in people:
    face = face_detector.detect(person["crop"])

    if face is None:
        print(f"Person {person['id']} -> No Face")

    else:
        print(f"Person {person['id']}")

        print("Face Found")

        print("Bounding Box :", face["bbox"])

        print("Landmarks :", len(face["landmarks"]))

        print()
