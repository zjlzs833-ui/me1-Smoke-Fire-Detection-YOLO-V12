from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path("Smoke Fire.pt")
INPUT_VIDEO = Path("assets/test_video.mp4")
OUTPUT_VIDEO = Path("outputs/fire_output.mp4")

CONF_THRESHOLD = 0.25
IMAGE_SIZE = 640


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}. "
            "Please place a test video at assets/test_video.mp4."
        )

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create output video: {OUTPUT_VIDEO}")

    print("Starting video detection...")

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
        writer.write(annotated_frame)

    cap.release()
    writer.release()

    print("Video detection completed.")
    print(f"Output file: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
