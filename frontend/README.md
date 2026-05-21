# Frontend

React + Vite monitoring console for the smoke and fire detection platform.

## Run

```bash
cd frontend
npm install
npm.cmd run dev
```

PowerShell may block `npm.ps1`, so `npm.cmd` is the recommended Windows command.

The frontend calls the FastAPI backend at `http://127.0.0.1:8000` by default. Override it with:

```bash
set VITE_API_BASE=http://127.0.0.1:8000
```

## Features

- RTSP network camera detection preview.
- Image and video upload detection.
- Fire situation summary and response actions.
- Feishu alert test and send status.
- SQLite-backed fire history list.
