import cv2

from inference.pipeline import DetectionPipeline

image = cv2.imread("assets/sample_images/single/image1.jpeg")

pipeline = DetectionPipeline()

results = pipeline.process(image)

print("=" * 60)

for person in results:
    print(f"Person ID      : {person['person_id']}")
    print(f"Sleep Status   : {person['sleep_status']}")
    print(f"EAR            : {person['ear']}")
    print(f"Person Box     : {person['person_bbox']}")
    print(f"Face Box       : {person['face_bbox']}")
    print()
