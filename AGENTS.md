# AGENTS.md

## 项目概览

本项目是一个基于 Ultralytics YOLO/YOLOv12 风格流程的烟雾与火灾检测项目。项目保留原始 Notebook，同时提供 Windows 本地可运行的 Python 推理、训练脚本，以及一个用于演示训练权重检测能力的 Web 网站。

默认推理权重文件位于项目根目录：

```text
Smoke Fire.pt
```

不要删除、移动或重命名该权重文件，除非同步修改所有推理入口和后端服务中的模型路径。

## 主要文件与目录

- `README.md`：原始项目说明。
- `REPRODUCE.md`：本地复现、环境安装、推理和训练说明。
- `WEB_DESIGN.md`：火灾智能检测网站设计思路与技术要求。
- `requirements.txt`：Python 依赖列表。
- `Smoke Fire.pt`：默认推理权重文件。
- `app.ipynb`：原始视频推理 Notebook。
- `smoke-fire-detection-yolo-v12.ipynb`：原始训练与数据处理 Notebook。
- `predict_image.py`：图片检测脚本。
- `predict_video.py`：视频检测脚本。
- `predict_camera.py`：摄像头实时检测脚本。
- `train_fire.py`：训练脚本。
- `backend/`：FastAPI 后端服务，提供 Web 检测接口。
- `frontend/`：React + Vite 前端网站。
- `assets/`：测试图片和测试视频输入目录。
- `outputs/`：推理输出目录。
- `datasets/fire_smoke/`：YOLO 格式训练数据集骨架。
- `Screenshots/`：原项目截图目录。

## 运行约定

- 推荐 conda 环境名：`yolov12_fire`。
- 推荐 Python 版本：`3.11`。
- 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

- 所有新增 Python 脚本都应从项目根目录运行。
- 新增脚本使用相对路径，不依赖作者本机绝对路径。
- 不要删除原始 Notebook 文件。
- 不要提交本地推理输出、训练输出、前端依赖目录或构建产物。

## 命令行推理入口

### 图片检测

`predict_image.py` 默认读取：

```text
assets/test.jpg
```

默认参数：

```text
MODEL_PATH = Smoke Fire.pt
CONF_THRESHOLD = 0.25
IMAGE_SIZE = 640
```

推理结果通常位于：

```text
runs/detect/predict
runs/detect/predict*
```

### 视频检测

`predict_video.py` 默认读取：

```text
assets/test_video.mp4
```

默认输出：

```text
outputs/fire_output.mp4
```

### 摄像头检测

`predict_camera.py` 默认使用摄像头索引：

```text
CAMERA_INDEX = 0
```

运行后会显示实时检测窗口，按 `q` 退出。

## Web 网站入口

### 后端

后端位于 `backend/`，使用 FastAPI 启动，并在启动时加载一次 `Smoke Fire.pt`。

运行命令：

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

主要接口：

- `GET /api/health`：检查模型加载状态和运行设备。
- `POST /api/detect/image`：上传图片并返回标注图、检测框、类别、置信度和告警等级。
- `POST /api/detect/video`：上传视频并返回标注视频、检测统计和告警等级。

Web 输出文件保存在：

```text
outputs/web/
```

该目录是本地运行产物，不应提交到源码仓库。

### 前端

前端位于 `frontend/`，使用 React + Vite。

首次运行：

```bash
cd frontend
npm install
npm run dev
```

如果 PowerShell 阻止 `npm.ps1`，在 Windows 上使用：

```bash
npm.cmd run dev
```

默认访问地址：

```text
http://127.0.0.1:5173
```

前端默认调用：

```text
http://127.0.0.1:8000
```

轮播图资源位于：

```text
frontend/public/banners/
```

## 数据集格式

默认数据集根目录：

```text
datasets/fire_smoke
```

图片目录：

```text
datasets/fire_smoke/images/train
datasets/fire_smoke/images/val
datasets/fire_smoke/images/test
```

标签目录：

```text
datasets/fire_smoke/labels/train
datasets/fire_smoke/labels/val
datasets/fire_smoke/labels/test
```

`datasets/fire_smoke/data.yaml` 当前类别顺序为：

```yaml
names:
  0: smoke
  1: fire
```

标签文件使用 YOLO 检测格式，每行一个目标：

```text
class_id x_center y_center width height
```

坐标应为相对于图片宽高归一化后的数值。

## 依赖

主要 Python 依赖包括：

- `ultralytics`
- `torch`
- `torchvision`
- `torchaudio`
- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `jupyter`
- `notebook`
- `fastapi`
- `uvicorn[standard]`
- `python-multipart`

前端依赖以 `frontend/package.json` 和 `frontend/package-lock.json` 为准。

## 提交与文件管理注意事项

- `runs/` 不应作为源码提交。
- `outputs/*.mp4`、`outputs/*.avi` 和 `outputs/web/` 是本地运行产物，不应提交。
- `frontend/node_modules/` 和 `frontend/dist/` 不应提交。
- `assets/test.jpg` 和 `assets/test_video.mp4` 可用于本地验证，但不是必须提交的源码。
- 原始 Notebook 和 `Smoke Fire.pt` 是项目复现的重要文件，应保持保留。
