from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

from backend.config.fusion_weights import validate_fusion_weights
from backend.services.fusion_service import calculate_fusion_risk
from backend.services.sensor_mock_service import SensorMockService


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "Smoke Fire.pt"
MEDIA_DIR = ROOT_DIR / "outputs" / "web"
UPLOAD_DIR = MEDIA_DIR / "uploads"
RESULT_DIR = MEDIA_DIR / "results"
SNAPSHOT_DIR = MEDIA_DIR / "snapshots"
DATA_DIR = MEDIA_DIR / "data"
DB_PATH = DATA_DIR / "fire_events.sqlite3"

CLASS_NAMES = {
    0: "smoke",
    1: "fire",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_ACTIONS = [
    "立即通过现场画面或人员巡查核实火点位置，避免误报后继续扩大响应。",
    "通知现场人员撤离到安全区域，优先保障人员安全。",
    "在确保安全的前提下切断相关区域电源、气源和易燃物来源。",
    "初起火灾可使用就近灭火器材处置，火势扩大时立即拨打 119。",
    "保留检测截图和视频记录，便于后续复盘与责任追踪。",
]


app = FastAPI(title="Smoke Fire Detection API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for directory in (MEDIA_DIR, UPLOAD_DIR, RESULT_DIR, SNAPSHOT_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

model: YOLO | None = None
model_error: str | None = None
model_lock = threading.Lock()
inference_device = "cuda" if torch.cuda.is_available() else "cpu"

streams: dict[str, "StreamState"] = {}
streams_lock = threading.Lock()

sensor_mock_service = SensorMockService()
latest_fire_confidence = 0.0
latest_fire_confidence_source = "未检测"
latest_fire_confidence_time: str | None = None
latest_fire_confidence_lock = threading.Lock()
fusion_alarm_threshold = 0.70
fusion_last_record_time = 0.0
fusion_record_lock = threading.Lock()
FUSION_HISTORY_COOLDOWN_SECONDS = 60


def load_env_files() -> None:
    for env_path in (ROOT_DIR / ".env", ROOT_DIR / "backend" / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class StreamCreate(BaseModel):
    name: str = "RTSP 摄像头"
    rtsp_url: str
    conf: float = 0.25
    imgsz: int = 640
    alert_cooldown_seconds: int = 300


class FeishuTestRequest(BaseModel):
    message: str = "飞书机器人连接测试成功。"


class SensorModeRequest(BaseModel):
    mode: str


class FusionThresholdRequest(BaseModel):
    threshold: float


class StreamState:
    def __init__(self, payload: StreamCreate) -> None:
        self.id = uuid.uuid4().hex
        self.name = payload.name.strip() or "RTSP 摄像头"
        self.rtsp_url = payload.rtsp_url.strip()
        self.conf = payload.conf
        self.imgsz = payload.imgsz
        self.alert_cooldown_seconds = max(30, payload.alert_cooldown_seconds)
        self.created_at = now_iso()
        self.updated_at = self.created_at
        self.status = "created"
        self.last_error: str | None = None
        self.frame_count = 0
        self.latest_jpeg: bytes | None = None
        self.last_summary: dict[str, Any] | None = None
        self.last_event: dict[str, Any] | None = None
        self.last_feishu_status = "not_sent"
        self.last_alert_time = 0.0
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=run_stream_worker, args=(self,), daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            if self.status not in {"stopped", "error"}:
                self.status = "stopping"
                self.updated_at = now_iso()

    def to_dict(self, include_url: bool = False) -> dict[str, Any]:
        with self.lock:
            data = {
                "id": self.id,
                "name": self.name,
                "status": self.status,
                "last_error": self.last_error,
                "conf": self.conf,
                "imgsz": self.imgsz,
                "alert_cooldown_seconds": self.alert_cooldown_seconds,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "frame_count": self.frame_count,
                "has_frame": self.latest_jpeg is not None,
                "mjpeg_url": f"/api/streams/{self.id}/mjpeg",
                "last_summary": self.last_summary,
                "last_event": self.last_event,
                "last_feishu_status": self.last_feishu_status,
            }
            if include_url:
                data["rtsp_url"] = self.rtsp_url
            return data


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def device_info() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }


def update_latest_fire_confidence(confidence: float, source: str) -> None:
    global latest_fire_confidence, latest_fire_confidence_source, latest_fire_confidence_time
    with latest_fire_confidence_lock:
        latest_fire_confidence = max(0.0, min(1.0, float(confidence)))
        latest_fire_confidence_source = source
        latest_fire_confidence_time = now_iso()


def get_latest_fire_confidence() -> dict[str, Any]:
    with latest_fire_confidence_lock:
        return {
            "fire_confidence": round(latest_fire_confidence, 4),
            "source": latest_fire_confidence_source,
            "updated_at": latest_fire_confidence_time,
        }


@app.on_event("startup")
def startup() -> None:
    load_env_files()
    validate_fusion_weights()
    init_db()
    load_model()


@app.on_event("shutdown")
def shutdown() -> None:
    with streams_lock:
        active_streams = list(streams.values())
    for stream in active_streams:
        stream.stop()
        stream.thread.join(timeout=2)


def load_model() -> None:
    global model, model_error, inference_device

    inference_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not MODEL_PATH.exists():
        model = None
        model_error = f"找不到模型文件: {MODEL_PATH}"
        return

    try:
        loaded_model = YOLO(str(MODEL_PATH))
        loaded_model.to(inference_device)
        model = loaded_model
        model_error = None
    except Exception as exc:  # pragma: no cover - depends on local model runtime
        model = None
        model_error = f"模型加载失败: {exc}"


def ensure_model() -> YOLO:
    if model is None and MODEL_PATH.exists():
        load_model()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_error or f"模型未加载。请确认 {MODEL_PATH.name} 位于项目根目录。",
        )
    return model


def ensure_model_for_worker() -> YOLO:
    if model is None and MODEL_PATH.exists():
        load_model()
    if model is None:
        raise RuntimeError(model_error or f"模型未加载: {MODEL_PATH.name}")
    return model


def init_db() -> None:
    with open_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                alert_level TEXT NOT NULL,
                decision TEXT NOT NULL,
                situation TEXT NOT NULL,
                message TEXT NOT NULL,
                actions_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                detections_json TEXT NOT NULL,
                media_url TEXT,
                source_url TEXT,
                feishu_status TEXT NOT NULL,
                feishu_error TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_alert_level ON events(alert_level)"
        )


def open_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA journal_mode=PERSIST")
    connection.execute("PRAGMA busy_timeout=3000")
    return connection


def media_url(path: Path | None) -> str | None:
    if path is None:
        return None
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
        raise HTTPException(status_code=400, detail=f"不支持的文件类型。允许类型: {allowed}")

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with target.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"上传文件保存失败: {exc}") from exc
    return target


def parse_detections(result: Any, frame: int | None = None) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        class_name = CLASS_NAMES.get(class_id, result.names.get(class_id, str(class_id)))
        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bbox": xyxy,
            "alert": class_name == "fire",
        }
        if frame is not None:
            detection["frame"] = frame
        detections.append(detection)
    return detections


def max_fire_confidence(detections: list[dict[str, Any]]) -> float:
    return max(
        (item["confidence"] for item in detections if item["class_name"] == "fire"),
        default=0.0,
    )


def summarize(
    detections: list[dict[str, Any]],
    frame_count: int | None = None,
    source_type: str = "image",
    source_name: str = "",
) -> dict[str, Any]:
    fire_count = sum(1 for item in detections if item["class_name"] == "fire")
    smoke_count = sum(1 for item in detections if item["class_name"] == "smoke")
    max_confidence = max((item["confidence"] for item in detections), default=0.0)
    fire_occurred = fire_count > 0
    smoke_detected = smoke_count > 0
    source_label = source_name or source_type

    if fire_occurred:
        alert_level = "danger"
        decision = "发生火灾"
        situation = (
            f"{source_label} 检测到 {fire_count} 个火焰目标"
            f"{'，同时检测到 ' + str(smoke_count) + ' 个烟雾目标' if smoke_count else ''}。"
        )
        message = (
            f"系统判断存在火灾风险，最高置信度 {max_confidence * 100:.1f}%。"
            "请立即核查现场并启动应急处置。"
        )
    elif smoke_detected:
        alert_level = "warning"
        decision = "疑似火灾风险"
        situation = f"{source_label} 检测到 {smoke_count} 个烟雾目标，暂未检测到火焰。"
        message = (
            f"系统判断存在烟雾预警，最高置信度 {max_confidence * 100:.1f}%。"
            "建议持续观察并安排人员巡查。"
        )
    else:
        alert_level = "safe"
        decision = "未发现火灾"
        situation = f"{source_label} 未检测到明显烟雾或火焰目标。"
        message = "当前画面未发现火灾迹象，建议保持监控。"

    summary = {
        "message": message,
        "situation": situation,
        "actions": DEFAULT_ACTIONS if fire_occurred else DEFAULT_ACTIONS[:2],
        "fire_count": fire_count,
        "smoke_count": smoke_count,
        "total_detections": len(detections),
        "max_confidence": round(max_confidence, 4),
        "frame_count": frame_count,
        "source_type": source_type,
        "source_name": source_name,
    }
    return {
        "alert_level": alert_level,
        "fire_occurred": fire_occurred,
        "smoke_detected": smoke_detected,
        "decision": decision,
        "summary": summary,
    }


def create_history_event(
    source_type: str,
    source_name: str,
    result_payload: dict[str, Any],
    detections: list[dict[str, Any]],
    media_path: Path | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    event_id = uuid.uuid4().hex
    created_at = now_iso()
    summary = result_payload["summary"]
    alert_level = result_payload["alert_level"]
    feishu_status = "pending" if alert_level == "danger" else "skipped"
    event = {
        "id": event_id,
        "created_at": created_at,
        "source_type": source_type,
        "source_name": source_name,
        "alert_level": alert_level,
        "decision": result_payload["decision"],
        "situation": summary["situation"],
        "message": summary["message"],
        "actions": summary["actions"],
        "summary": summary,
        "detections": detections[:300],
        "media_url": media_url(media_path),
        "source_url": media_url(source_path),
        "feishu_status": feishu_status,
        "feishu_error": None,
    }

    with open_db() as connection:
        connection.execute(
            """
            INSERT INTO events (
                id, created_at, source_type, source_name, alert_level, decision,
                situation, message, actions_json, summary_json, detections_json,
                media_url, source_url, feishu_status, feishu_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["id"],
                event["created_at"],
                event["source_type"],
                event["source_name"],
                event["alert_level"],
                event["decision"],
                event["situation"],
                event["message"],
                json.dumps(event["actions"], ensure_ascii=False),
                json.dumps(event["summary"], ensure_ascii=False),
                json.dumps(event["detections"], ensure_ascii=False),
                event["media_url"],
                event["source_url"],
                event["feishu_status"],
                event["feishu_error"],
            ),
        )

    if alert_level == "danger":
        status, error = send_feishu_alert(event)
        event["feishu_status"] = status
        event["feishu_error"] = error
        update_event_feishu_status(event_id, status, error)

    return event


def update_event_feishu_status(event_id: str, status: str, error: str | None) -> None:
    with open_db() as connection:
        connection.execute(
            "UPDATE events SET feishu_status = ?, feishu_error = ? WHERE id = ?",
            (status, error, event_id),
        )


def row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "source_type": row["source_type"],
        "source_name": row["source_name"],
        "alert_level": row["alert_level"],
        "decision": row["decision"],
        "situation": row["situation"],
        "message": row["message"],
        "actions": json.loads(row["actions_json"]),
        "summary": json.loads(row["summary_json"]),
        "detections": json.loads(row["detections_json"]),
        "media_url": row["media_url"],
        "source_url": row["source_url"],
        "feishu_status": row["feishu_status"],
        "feishu_error": row["feishu_error"],
    }


def feishu_config() -> tuple[str | None, str | None]:
    webhook = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    secret = os.getenv("FEISHU_WEBHOOK_SECRET", "").strip()
    return webhook or None, secret or None


def feishu_sign(secret: str, timestamp: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def send_feishu_payload(payload: dict[str, Any]) -> tuple[str, str | None]:
    webhook, secret = feishu_config()
    if not webhook:
        return "disabled", "未配置 FEISHU_WEBHOOK_URL"

    if secret:
        timestamp = str(int(time.time()))
        payload = {**payload, "timestamp": timestamp, "sign": feishu_sign(secret, timestamp)}

    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(
        webhook,
        data=request_body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
    try:
        with opener.open(request, timeout=8) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                return "failed", response_body
            try:
                payload_response = json.loads(response_body)
            except json.JSONDecodeError:
                return "sent", None
            if payload_response.get("code", 0) != 0:
                return "failed", payload_response.get("msg", response_body)
            return "sent", None
    except urlerror.URLError as exc:
        return "failed", str(exc)


def feishu_card_for_event(event: dict[str, Any]) -> dict[str, Any]:
    template = {
        "danger": "red",
        "warning": "orange",
        "safe": "green",
    }.get(event["alert_level"], "blue")
    if event.get("source_type") == "fusion":
        summary = event["summary"]
        sensor = summary.get("sensor_data", {})
        title = "火灾检测系统 - 多传感器融合报警"
        content = (
            f"**报警等级**\n{event['decision']}\n\n"
            f"**综合风险值**\n{summary.get('risk_score', 0):.3f}\n\n"
            f"**YOLO置信度**\n{summary.get('fire_confidence', 0):.3f}\n\n"
            f"**传感器数据**\n"
            f"- 温度: {sensor.get('temperature', 0)} °C\n"
            f"- 烟雾浓度: {sensor.get('smoke', 0)} ppm\n"
            f"- CO浓度: {sensor.get('co', 0)} ppm\n"
            f"- 火焰状态: {'检测到火焰' if sensor.get('flame') else '未检测到火焰'}\n\n"
            f"**报警时间**\n{event['created_at']}\n\n"
            f"**历史记录 ID**\n{event['id']}"
        )
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "template": template,
                    "title": {"tag": "plain_text", "content": title},
                },
                "elements": [
                    {"tag": "markdown", "content": content},
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"来源: {event['source_name']} | 时间: {event['created_at']}",
                            }
                        ],
                    },
                ],
            },
        }

    title = {
        "danger": "火灾检测系统 - 火灾报警",
        "warning": "火灾检测系统 - 烟雾预警",
        "safe": "火灾检测系统 - 火情研判",
    }.get(event["alert_level"], "火灾检测系统 - 火情研判")
    actions = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(event["actions"]))
    detections = event["summary"]
    content = (
        f"**火情说明**\n{event['situation']}\n\n"
        f"**系统判断**\n{event['message']}\n\n"
        f"**检测统计**\n"
        f"- 火焰目标: {detections.get('fire_count', 0)}\n"
        f"- 烟雾目标: {detections.get('smoke_count', 0)}\n"
        f"- 最高置信度: {detections.get('max_confidence', 0) * 100:.1f}%\n\n"
        f"**应对措施**\n{actions}\n\n"
        f"**历史记录 ID**\n{event['id']}"
    )
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "template": template,
                "title": {"tag": "plain_text", "content": title},
            },
            "elements": [
                {"tag": "markdown", "content": content},
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": f"来源: {event['source_name']} | 时间: {event['created_at']}",
                        }
                    ],
                },
            ],
        },
    }


def send_feishu_alert(event: dict[str, Any]) -> tuple[str, str | None]:
    return send_feishu_payload(feishu_card_for_event(event))


def predict_frame(yolo: YOLO, frame: Any, conf: float, imgsz: int) -> Any:
    with model_lock:
        results = yolo.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            device=inference_device,
            verbose=False,
        )
    return results[0]


def run_stream_worker(stream: StreamState) -> None:
    capture: cv2.VideoCapture | None = None
    try:
        yolo = ensure_model_for_worker()
        with stream.lock:
            stream.status = "connecting"
            stream.updated_at = now_iso()

        capture = cv2.VideoCapture(stream.rtsp_url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not capture.isOpened():
            raise RuntimeError("无法打开 RTSP 视频流，请检查地址、账号密码和网络。")

        with stream.lock:
            stream.status = "running"
            stream.last_error = None
            stream.updated_at = now_iso()

        while not stream.stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                with stream.lock:
                    stream.status = "reconnecting"
                    stream.last_error = "读取视频流失败，正在尝试重连。"
                    stream.updated_at = now_iso()
                capture.release()
                time.sleep(1.5)
                capture = cv2.VideoCapture(stream.rtsp_url, cv2.CAP_FFMPEG)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                continue

            result = predict_frame(yolo, frame, stream.conf, stream.imgsz)
            with stream.lock:
                stream.frame_count += 1
                frame_number = stream.frame_count

            detections = parse_detections(result, frame=frame_number)
            summary = summarize(
                detections,
                frame_count=frame_number,
                source_type="rtsp",
                source_name=stream.name,
            )
            update_latest_fire_confidence(
                max_fire_confidence(detections),
                f"RTSP: {stream.name}",
            )
            annotated = result.plot()
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82],
            )
            if encoded_ok:
                with stream.lock:
                    stream.latest_jpeg = encoded.tobytes()
                    stream.last_summary = summary
                    stream.status = "running"
                    stream.last_error = None
                    stream.updated_at = now_iso()

            if summary["fire_occurred"]:
                current_time = time.time()
                if current_time - stream.last_alert_time >= stream.alert_cooldown_seconds:
                    snapshot_path = SNAPSHOT_DIR / f"{stream.id}_{frame_number}.jpg"
                    cv2.imwrite(str(snapshot_path), annotated)
                    event = create_history_event(
                        source_type="rtsp",
                        source_name=stream.name,
                        result_payload=summary,
                        detections=detections,
                        media_path=snapshot_path,
                    )
                    with stream.lock:
                        stream.last_alert_time = current_time
                        stream.last_event = event
                        stream.last_feishu_status = event["feishu_status"]

            time.sleep(0.03)

        with stream.lock:
            stream.status = "stopped"
            stream.updated_at = now_iso()
    except Exception as exc:  # pragma: no cover - depends on camera/network
        with stream.lock:
            stream.status = "error"
            stream.last_error = str(exc)
            stream.updated_at = now_iso()
    finally:
        if capture is not None:
            capture.release()


def mjpeg_generator(stream: StreamState):
    while not stream.stop_event.is_set():
        with stream.lock:
            frame = stream.latest_jpeg
            status = stream.status
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        elif status in {"stopped", "error"}:
            break
        time.sleep(0.1)


@app.get("/api/health")
def health() -> dict[str, Any]:
    info = device_info()
    webhook, secret = feishu_config()
    with streams_lock:
        active_streams = sum(1 for item in streams.values() if item.thread.is_alive())

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
        "history_db": str(DB_PATH.relative_to(ROOT_DIR)),
        "feishu_configured": webhook is not None,
        "feishu_signed": secret is not None,
        "active_streams": active_streams,
        **info,
    }


def build_fusion_event_payload(fusion: dict[str, Any]) -> dict[str, Any]:
    sensor = fusion["sensor_data"]
    risk_score = fusion["risk_score"]
    risk_level = fusion["risk_level"]
    fire_confidence = fusion["fire_confidence"]
    flame_text = "检测到火焰" if sensor.get("flame") else "未检测到火焰"
    situation = (
        f"多传感器融合判断为{risk_level}，综合风险值 {risk_score:.3f}。"
        f"温度 {sensor.get('temperature')} °C，烟雾 {sensor.get('smoke')} ppm，"
        f"CO {sensor.get('co')} ppm，火焰状态：{flame_text}。"
    )
    message = (
        f"YOLO 最高置信度 {fire_confidence * 100:.1f}%，"
        f"融合报警阈值 {fusion['alarm_threshold']:.2f}。"
    )
    actions = DEFAULT_ACTIONS if fusion["alarm"] else DEFAULT_ACTIONS[:2]
    return {
        "alert_level": fusion["alert_level"],
        "decision": risk_level,
        "fire_occurred": fusion["alarm"],
        "smoke_detected": sensor.get("smoke", 0) >= 150,
        "summary": {
            "message": message,
            "situation": situation,
            "actions": actions,
            "fire_count": 1 if fusion["alarm"] else 0,
            "smoke_count": 1 if sensor.get("smoke", 0) >= 150 else 0,
            "total_detections": 0,
            "max_confidence": fire_confidence,
            "frame_count": None,
            "source_type": "fusion",
            "source_name": "多传感器融合",
            "fire_confidence": fire_confidence,
            "sensor_data": sensor,
            "risk_detail": fusion["risk_detail"],
            "weight_detail": fusion["weight_detail"],
            "risk_score": risk_score,
            "risk_level": risk_level,
            "alarm": fusion["alarm"],
            "alarm_threshold": fusion["alarm_threshold"],
        },
    }


def maybe_record_fusion_event(fusion: dict[str, Any]) -> dict[str, Any] | None:
    global fusion_last_record_time
    if fusion["alert_level"] == "safe":
        return None

    current_time = time.time()
    with fusion_record_lock:
        if current_time - fusion_last_record_time < FUSION_HISTORY_COOLDOWN_SECONDS:
            return None
        fusion_last_record_time = current_time

    return create_history_event(
        source_type="fusion",
        source_name="多传感器融合",
        result_payload=build_fusion_event_payload(fusion),
        detections=[],
    )


@app.get("/api/sensors/latest")
def get_latest_sensors() -> dict[str, Any]:
    return sensor_mock_service.latest()


@app.post("/api/sensors/mode")
def set_sensor_mode(payload: SensorModeRequest) -> dict[str, Any]:
    try:
        return sensor_mock_service.set_mode(payload.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/fusion/threshold")
def set_fusion_threshold(payload: FusionThresholdRequest) -> dict[str, Any]:
    global fusion_alarm_threshold
    if not 0.1 <= payload.threshold <= 0.95:
        raise HTTPException(status_code=400, detail="threshold 必须在 0.1 到 0.95 之间")
    fusion_alarm_threshold = round(float(payload.threshold), 2)
    return {"success": True, "threshold": fusion_alarm_threshold}


@app.get("/api/fusion/status")
def get_fusion_status() -> dict[str, Any]:
    sensor_data = sensor_mock_service.latest()
    yolo_state = get_latest_fire_confidence()
    fusion = calculate_fusion_risk(
        fire_confidence=yolo_state["fire_confidence"],
        temperature=sensor_data["temperature"],
        smoke=sensor_data["smoke"],
        co=sensor_data["co"],
        flame=sensor_data["flame"],
        alarm_threshold=fusion_alarm_threshold,
    )
    fusion["sensor_data"] = {**fusion["sensor_data"], "mode": sensor_data["mode"]}
    fusion["timestamp"] = sensor_data["timestamp"]
    fusion["yolo_source"] = yolo_state["source"]
    fusion["yolo_updated_at"] = yolo_state["updated_at"]
    fusion["history_event"] = maybe_record_fusion_event(fusion)
    return fusion


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
        result = predict_frame(yolo, str(input_path), conf, imgsz)
        annotated = result.plot()
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError("无法写入标注图片")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"图片模型推理失败: {exc}") from exc

    detections = parse_detections(result)
    summary = summarize(detections, source_type="image", source_name=file.filename or "上传图片")
    update_latest_fire_confidence(max_fire_confidence(detections), f"图片: {file.filename or input_path.name}")
    event = create_history_event(
        "image",
        file.filename or input_path.name,
        summary,
        detections,
        media_path=output_path,
        source_path=input_path,
    )
    return {
        "result_image_url": media_url(output_path),
        "source_image_url": media_url(input_path),
        "detections": detections,
        "history_event": event,
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

            frame_count += 1
            result = predict_frame(yolo, frame, conf, imgsz)
            writer.write(result.plot())
            all_detections.extend(parse_detections(result, frame=frame_count))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"视频模型推理失败: {exc}") from exc
    finally:
        capture.release()
        writer.release()

    summary = summarize(
        all_detections,
        frame_count=frame_count,
        source_type="video",
        source_name=file.filename or "上传视频",
    )
    update_latest_fire_confidence(max_fire_confidence(all_detections), f"视频: {file.filename or input_path.name}")
    event = create_history_event(
        "video",
        file.filename or input_path.name,
        summary,
        all_detections,
        media_path=output_path,
        source_path=input_path,
    )
    return {
        "result_video_url": media_url(output_path),
        "source_video_url": media_url(input_path),
        "frame_count": frame_count,
        "detections_count": len(all_detections),
        "detections": all_detections[:300],
        "history_event": event,
        "inference_device": inference_device,
        **summary,
    }


@app.post("/api/streams")
def create_stream(payload: StreamCreate) -> dict[str, Any]:
    ensure_model()
    conf, imgsz = validate_thresholds(payload.conf, payload.imgsz)
    payload.conf = conf
    payload.imgsz = imgsz
    if not payload.rtsp_url.strip():
        raise HTTPException(status_code=400, detail="rtsp_url 不能为空")
    stream = StreamState(payload)
    with streams_lock:
        active_streams = sum(
            1 for item in streams.values() if item.status not in {"stopped", "error", "offline"}
        )
        if active_streams >= 3:
            raise HTTPException(status_code=400, detail="最多同时接入 3 路摄像头")
        streams[stream.id] = stream
    stream.start()
    return stream.to_dict(include_url=True)


@app.get("/api/streams")
def list_streams() -> dict[str, Any]:
    with streams_lock:
        items = [stream.to_dict() for stream in streams.values()]
    return {"items": items, "total": len(items)}


@app.get("/api/streams/{stream_id}")
def get_stream(stream_id: str) -> dict[str, Any]:
    stream = get_stream_or_404(stream_id)
    return stream.to_dict(include_url=True)


@app.post("/api/streams/{stream_id}/stop")
def stop_stream(stream_id: str) -> dict[str, Any]:
    stream = get_stream_or_404(stream_id)
    stream.stop()
    stream.thread.join(timeout=3)
    return stream.to_dict(include_url=True)


@app.get("/api/streams/{stream_id}/mjpeg")
def stream_mjpeg(stream_id: str) -> StreamingResponse:
    stream = get_stream_or_404(stream_id)
    return StreamingResponse(
        mjpeg_generator(stream),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def get_stream_or_404(stream_id: str) -> StreamState:
    with streams_lock:
        stream = streams.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="找不到实时检测任务")
    return stream


@app.get("/api/history")
def list_history(
    limit: int = 20,
    offset: int = 0,
    source_type: str | None = None,
    alert_level: str | None = None,
) -> dict[str, Any]:
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)
    conditions: list[str] = []
    params: list[Any] = []
    if source_type:
        conditions.append("source_type = ?")
        params.append(source_type)
    if alert_level:
        conditions.append("alert_level = ?")
        params.append(alert_level)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with open_db() as connection:
        connection.row_factory = sqlite3.Row
        total = connection.execute(
            f"SELECT COUNT(*) FROM events {where}",
            params,
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT * FROM events {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
    return {"items": [row_to_event(row) for row in rows], "total": total}


@app.get("/api/history/{event_id}")
def get_history_event(event_id: str) -> dict[str, Any]:
    return read_history_event(event_id)


def read_history_event(event_id: str) -> dict[str, Any]:
    with open_db() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="找不到历史记录")
    return row_to_event(row)


@app.post("/api/history/{event_id}/feishu")
def sync_history_event_to_feishu(event_id: str) -> dict[str, Any]:
    event = read_history_event(event_id)
    status, error = send_feishu_alert(event)
    update_event_feishu_status(event_id, status, error)
    event["feishu_status"] = status
    event["feishu_error"] = error
    return {"status": status, "error": error, "event": event}


@app.post("/api/feishu/test")
def test_feishu(payload: FeishuTestRequest) -> dict[str, Any]:
    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "template": "blue",
                "title": {"tag": "plain_text", "content": "火灾检测系统测试"},
            },
            "elements": [
                {"tag": "markdown", "content": payload.message},
                {
                    "tag": "note",
                    "elements": [{"tag": "plain_text", "content": f"发送时间: {now_iso()}"}],
                },
            ],
        },
    }
    status, error = send_feishu_payload(card)
    return {
        "configured": feishu_config()[0] is not None,
        "status": status,
        "error": error,
    }
