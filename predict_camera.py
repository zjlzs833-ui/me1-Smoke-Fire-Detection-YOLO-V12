from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path("Smoke Fire.pt")

CONF_THRESHOLD = 0.25
IMAGE_SIZE = 640
CAMERA_INDEX = 0


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open camera. Please check Windows camera permission "
            "or whether another program is using the camera."
        )

    print("Camera detection started. Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        annotated_frame = results[0].plot()
        cv2.imshow("Smoke and Fire Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
