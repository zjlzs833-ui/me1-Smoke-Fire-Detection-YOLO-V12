# Backend

FastAPI service for smoke/fire detection, RTSP real-time monitoring, Feishu alerting, and local SQLite history.

## Python 3.11 Run

Use the project conda environment instead of the system Python:

```bash
conda activate yolov12_fire
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Or run without activating:

```bash
conda run -n yolov12_fire uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The service expects `Smoke Fire.pt` in the project root.

## Feishu Alerts

Configure a Feishu group custom bot before starting the backend:

```powershell
$env:FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"
$env:FEISHU_WEBHOOK_SECRET="optional-sign-secret"
```

`FEISHU_WEBHOOK_SECRET` is optional and is only needed when the bot enables signature verification.

You can also create `.env` in the project root or `backend/.env`:

```text
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxxx
FEISHU_WEBHOOK_SECRET=optional-sign-secret
```

## Main APIs

- `GET /api/health`
- `POST /api/detect/image`
- `POST /api/detect/video`
- `POST /api/streams`
- `GET /api/streams/{stream_id}`
- `GET /api/streams/{stream_id}/mjpeg`
- `POST /api/streams/{stream_id}/stop`
- `GET /api/history`
- `GET /api/history/{event_id}`
- `POST /api/history/{event_id}/feishu`
- `POST /api/feishu/test`

Runtime media is saved under `outputs/web/`, and the SQLite database is saved at `outputs/web/data/fire_events.sqlite3`.
