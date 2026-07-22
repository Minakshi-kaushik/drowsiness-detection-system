import cv2

from inference.detect_person import PersonDetector


detector = PersonDetector()

image = cv2.imread("assets/sample_images/test.jpg")

people = detector.detect(image)

print()

print("=" * 40)

print("Detected People")

print("=" * 40)

for person in people:
    print(person)

print()

print("Total People :", len(people))
