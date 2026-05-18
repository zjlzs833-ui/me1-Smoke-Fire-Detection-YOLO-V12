from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path("Smoke Fire.pt")
IMAGE_PATH = Path("assets/test.jpg")

CONF_THRESHOLD = 0.25
IMAGE_SIZE = 640


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}. "
            "Please place a test image at assets/test.jpg."
        )

    model = YOLO(str(MODEL_PATH))
    model.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        save=True,
    )

    print("Image detection completed.")
    print("Results are usually saved in runs/detect/predict or runs/detect/predict*.")


if __name__ == "__main__":
    main()
