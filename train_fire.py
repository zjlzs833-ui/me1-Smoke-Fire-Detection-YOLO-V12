from pathlib import Path

from ultralytics import YOLO


DATA_YAML = Path("datasets/fire_smoke/data.yaml")
INITIAL_MODEL = "yolo12n.pt"

EPOCHS = 60
IMAGE_SIZE = 640
BATCH_SIZE = 16


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            "Dataset config file not found: datasets/fire_smoke/data.yaml. "
            "Please prepare the YOLO-format dataset structure described in REPRODUCE.md."
        )

    model = YOLO(INITIAL_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        verbose=True,
    )

    print("Training completed.")
    print("Results are usually saved in runs/detect/train or runs/detect/train*.")
    print("Best weights are usually saved at runs/detect/train/weights/best.pt.")


if __name__ == "__main__":
    main()
