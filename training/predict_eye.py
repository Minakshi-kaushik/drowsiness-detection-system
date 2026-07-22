import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "models/eye/eye_state_model.keras"

CLASS_NAMES = ["awake", "sleepy"]


class EyeStatePredictor:
    def __init__(self):

        self.model = tf.keras.models.load_model(MODEL_PATH)

    def predict(self, eye_image):

        if eye_image is None:
            return None, 0.0

        if eye_image.size == 0:
            return None, 0.0

        eye = cv2.resize(eye_image, (64, 64))

        eye = eye.astype("float32") / 255.0

        eye = np.expand_dims(eye, axis=0)

        predictions = self.model.predict(eye, verbose=0)

        index = np.argmax(predictions)

        confidence = float(predictions[0][index])

        return CLASS_NAMES[index], confidence
