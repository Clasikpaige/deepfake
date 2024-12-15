import cv2
import tensorflow as tf
import numpy as np
import pyvirtualcam

MODEL_PATH = "models/deepfake_model/deepfake_model.h5"

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    return frame

def live_face_swap():
    model = load_model()
    cap = cv2.VideoCapture(0)

    with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = preprocess_frame(frame)
            swapped_frame = model.predict(processed_frame)[0]
            swapped_frame = (swapped_frame * 255).astype("uint8")
            swapped_frame = cv2.resize(swapped_frame, (640, 480))

            cam.send(swapped_frame)
            cam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_face_swap()
