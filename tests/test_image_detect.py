import cv2

from inference.detect_image import ImageDetector

detector = ImageDetector()

image, people, output = detector.process("assets/sample_images/test.jpg")

print("=" * 40)
print("Detection Successful")
print("=" * 40)

print("People Detected :", len(people))
print("Output Saved :", output)

cv2.imshow("Result", image)

cv2.waitKey(0)
# to remove the present windows
cv2.destroyAllWindows()
