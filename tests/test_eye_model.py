import cv2

from training.predict_eye import EyeStatePredictor


IMAGE_PATH = "datasets/test/awake/0.jpg"  # change to any image


predictor = EyeStatePredictor()

image = cv2.imread(IMAGE_PATH)

label, confidence = predictor.predict(image)

print()

print("Prediction :", label)

print("Confidence :", round(confidence, 3))
