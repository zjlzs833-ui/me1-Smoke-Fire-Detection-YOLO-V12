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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

model: YOLO | None = None
model_error: str | None = None
inference_device = "cuda" if torch.cuda.is_available() else "cpu"


def device_info() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "torch_version": torch.__version__,
    }


@app.on_event("startup")
def load_model() -> None:
    global model, model_error, inference_device

    inference_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not MODEL_PATH.exists():
        model = None
        model_error = f"找不到模型文件：{MODEL_PATH}"
        return

    try:
        model = YOLO(str(MODEL_PATH))
        model.to(inference_device)
        model_error = None
    except Exception as exc:
        model = None
        model_error = f"模型加载失败：{exc}"


def ensure_model() -> YOLO:
    if model is None and MODEL_PATH.exists():
        load_model()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_error or f"模型未加载。请确认 {MODEL_PATH.name} 位于项目根目录。",
        )
    return model


def media_url(path: Path) -> str:
    relative = path.relative_to(MEDIA_DIR).as_posix()
    return f"/media/{relative}"


def validate_thresholds(conf: float, imgsz: int) -> tuple[float, int]:
    if not 0.01 <= conf <= 0.9:
        raise HTTPException(status_code=400, detail="conf 必须在 0.01 到 0.9 之间")
    if imgsz < 320 or imgsz > 1280:
        raise HTTPException(status_code=400, detail="imgsz 必须在 320 到 1280 之间")
    return conf, imgsz


async def save_upload(file: UploadFile, allowed_extensions: set[str]) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(status_code=400, detail=f"不支持的文件类型。允许类型：{allowed}")

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with target.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"上传文件保存失败：{exc}") from exc
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
        detections.append(
            {
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": xyxy,
                "alert": class_name == "fire",
            }
        )
    return detections


def summarize(detections: list[dict[str, Any]], frame_count: int | None = None) -> dict[str, Any]:
    fire_count = sum(1 for item in detections if item["class_name"] == "fire")
    smoke_count = sum(1 for item in detections if item["class_name"] == "smoke")
    max_confidence = max((item["confidence"] for item in detections), default=0)
    fire_occurred = fire_count > 0
    smoke_detected = smoke_count > 0

    if fire_occurred:
        alert_level = "danger"
        decision = "发生火灾"
        message = f"检测到 {fire_count} 个火焰目标，判断为发生火灾，请立即核查现场。"
    elif smoke_detected:
        alert_level = "warning"
        decision = "疑似火灾风险"
        message = f"检测到 {smoke_count} 个烟雾目标，暂未检测到火焰，建议持续观察。"
    else:
        alert_level = "safe"
        decision = "未发生火灾"
        message = "未检测到明显烟雾或火焰，当前图片未发现火灾迹象。"

    return {
        "alert_level": alert_level,
        "fire_occurred": fire_occurred,
        "smoke_detected": smoke_detected,
        "decision": decision,
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
    info = device_info()
    if model is not None:
        health_state = "ready"
        message = "后端可访问，模型已加载。"
    elif model_error:
        health_state = "model_error"
        message = model_error
    else:
        health_state = "loading"
        message = "后端可访问，模型正在加载或尚未加载。"

    return {
        "status": "ok" if model is not None else "model_missing",
        "health_state": health_state,
        "model_loaded": model is not None,
        "model_path": MODEL_PATH.name,
        "model_error": model_error,
        "message": message,
        "inference_device": inference_device,
        **info,
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

    try:
        results = yolo.predict(
            source=str(input_path),
            imgsz=imgsz,
            conf=conf,
            device=inference_device,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError("无法写入标注图片")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"图片模型推理失败：{exc}") from exc

    detections = parse_detections(result)
    summary = summarize(detections)
    return {
        "result_image_url": media_url(output_path),
        "source_image_url": media_url(input_path),
        "detections": detections,
        "inference_device": inference_device,
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
        raise HTTPException(status_code=400, detail="无法打开上传的视频文件")

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
        raise HTTPException(status_code=500, detail="无法创建输出视频文件")

    frame_count = 0
    all_detections: list[dict[str, Any]] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = yolo.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                device=inference_device,
                verbose=False,
            )
            result = results[0]
            writer.write(result.plot())
            frame_count += 1

            for detection in parse_detections(result):
                all_detections.append({"frame": frame_count, **detection})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"视频模型推理失败：{exc}") from exc
    finally:
        capture.release()
        writer.release()

    summary = summarize(all_detections, frame_count=frame_count)
    return {
        "result_video_url": media_url(output_path),
        "source_video_url": media_url(input_path),
        "frame_count": frame_count,
        "detections_count": len(all_detections),
        "detections": all_detections[:200],
        "inference_device": inference_device,
        **summary,
    }
