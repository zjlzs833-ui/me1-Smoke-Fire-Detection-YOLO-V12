from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "Smoke Fire.pt"
MEDIA_DIR = ROOT_DIR / "outputs" / "web"
UPLOAD_DIR = MEDIA_DIR / "uploads"
RESULT_DIR = MEDIA_DIR / "results"

CLASS_NAMES = {
    0: "smoke",
    1: "fire",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = FastAPI(title="Smoke Fire Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

model: YOLO | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        model = None
        return
    model = YOLO(str(MODEL_PATH))


def ensure_model() -> YOLO:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded. Expected weight file: {MODEL_PATH.name}",
        )
    return model


def media_url(path: Path) -> str:
    relative = path.relative_to(MEDIA_DIR).as_posix()
    return f"/media/{relative}"


def validate_thresholds(conf: float, imgsz: int) -> tuple[float, int]:
    if not 0.01 <= conf <= 0.9:
        raise HTTPException(status_code=400, detail="conf must be between 0.01 and 0.9")
    if imgsz < 320 or imgsz > 1280:
        raise HTTPException(status_code=400, detail="imgsz must be between 320 and 1280")
    return conf, imgsz


async def save_upload(file: UploadFile, allowed_extensions: set[str]) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return target


def parse_detections(result: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        class_name = CLASS_NAMES.get(class_id, result.names.get(class_id, str(class_id)))
        alert = class_name == "fire" or (class_name == "smoke" and confidence >= 0.5)
        detections.append(
            {
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": xyxy,
                "alert": alert,
            }
        )
    return detections


def summarize(detections: list[dict[str, Any]], frame_count: int | None = None) -> dict[str, Any]:
    fire_count = sum(1 for item in detections if item["class_name"] == "fire")
    smoke_count = sum(1 for item in detections if item["class_name"] == "smoke")
    max_confidence = max((item["confidence"] for item in detections), default=0)

    if fire_count:
        alert_level = "danger"
        message = f"检测到 {fire_count} 个火焰目标，建议立即核查现场。"
    elif smoke_count:
        alert_level = "warning"
        message = f"检测到 {smoke_count} 个烟雾目标，建议持续观察。"
    else:
        alert_level = "safe"
        message = "未检测到明显烟雾或火焰。"

    return {
        "alert_level": alert_level,
        "summary": {
            "message": message,
            "fire_count": fire_count,
            "smoke_count": smoke_count,
            "total_detections": len(detections),
            "max_confidence": round(max_confidence, 4),
            "frame_count": frame_count,
        },
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "model_missing",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH.name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
) -> dict[str, Any]:
    yolo = ensure_model()
    conf, imgsz = validate_thresholds(conf, imgsz)
    input_path = await save_upload(file, IMAGE_EXTENSIONS)
    output_path = RESULT_DIR / f"{input_path.stem}_detected.jpg"

    results = yolo.predict(source=str(input_path), imgsz=imgsz, conf=conf, verbose=False)
    result = results[0]
    annotated = result.plot()
    if not cv2.imwrite(str(output_path), annotated):
        raise HTTPException(status_code=500, detail="Failed to write annotated image")

    detections = parse_detections(result)
    summary = summarize(detections)
    return {
        "result_image_url": media_url(output_path),
        "source_image_url": media_url(input_path),
        "detections": detections,
        **summary,
    }


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
) -> dict[str, Any]:
    yolo = ensure_model()
    conf, imgsz = validate_thresholds(conf, imgsz)
    input_path = await save_upload(file, VIDEO_EXTENSIONS)
    output_path = RESULT_DIR / f"{input_path.stem}_detected.mp4"

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open uploaded video")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise HTTPException(status_code=500, detail="Unable to create output video")

    frame_count = 0
    all_detections: list[dict[str, Any]] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        results = yolo.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
        result = results[0]
        writer.write(result.plot())
        frame_count += 1

        for detection in parse_detections(result):
            all_detections.append({"frame": frame_count, **detection})

    capture.release()
    writer.release()

    summary = summarize(all_detections, frame_count=frame_count)
    return {
        "result_video_url": media_url(output_path),
        "source_video_url": media_url(input_path),
        "frame_count": frame_count,
        "detections_count": len(all_detections),
        "detections": all_detections[:200],
        **summary,
    }
