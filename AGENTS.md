# AGENTS.md

## 项目概览

本项目是一个基于 Ultralytics YOLO/YOLOv12 风格流程的烟雾与火灾检测项目。项目保留了原始 Notebook，同时整理了可在 Windows 本地直接运行的 Python 推理和训练脚本。

推理默认使用项目根目录下的权重文件：

```text
Smoke Fire.pt
```

## 主要文件与目录

- `README.md`：原始项目说明。
- `REPRODUCE.md`：本地复现、环境安装、推理和训练说明。
- `requirements.txt`：项目依赖列表。
- `Smoke Fire.pt`：默认推理权重文件。
- `app.ipynb`：原始视频推理 Notebook。
- `smoke-fire-detection-yolo-v12.ipynb`：原始训练与数据处理 Notebook。
- `predict_image.py`：图片检测脚本。
- `predict_video.py`：视频检测脚本。
- `predict_camera.py`：摄像头实时检测脚本。
- `train_fire.py`：训练脚本。
- `assets/`：测试图片和测试视频输入目录。
- `outputs/`：视频推理输出目录。
- `datasets/fire_smoke/`：YOLO 格式训练数据集骨架。
- `Screenshots/`：原项目截图目录。

## 运行约定

- 推荐 conda 环境名：`yolov12_fire`。
- 推荐 Python 版本：`3.11`。
- 安装依赖命令：

```bash
pip install -r requirements.txt
```

- 所有新增 Python 脚本都应从项目根目录运行。
- 新增脚本使用相对路径，不依赖作者本机绝对路径。
- 不要删除原始 Notebook 文件。
- 不要删除或移动默认权重文件 `Smoke Fire.pt`，除非同步修改脚本中的模型路径。

## 推理入口

### 图片检测

`predict_image.py` 默认读取：

```text
assets/test.jpg
```

默认使用：

```text
MODEL_PATH = Smoke Fire.pt
CONF_THRESHOLD = 0.25
IMAGE_SIZE = 640
```

推理结果由 Ultralytics 保存，通常位于：

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

默认使用：

```text
MODEL_PATH = Smoke Fire.pt
CONF_THRESHOLD = 0.25
IMAGE_SIZE = 640
```

### 摄像头检测

`predict_camera.py` 默认使用摄像头索引：

```text
CAMERA_INDEX = 0
```

运行后会显示实时检测窗口，按 `q` 退出。

### 推理参数优化经验

如果测试素材检测效果较弱，可以尝试降低置信度阈值并提高输入尺寸，例如：

```text
conf = 0.05
imgsz = 960
```

这类参数适合用于调试和观察低置信度检测结果。

## 训练入口

`train_fire.py` 使用 Ultralytics `YOLO` 训练接口，默认初始模型为：

```text
yolo12n.pt
```

默认数据集配置文件：

```text
datasets/fire_smoke/data.yaml
```

默认训练参数：

```text
epochs = 60
imgsz = 640
batch = 16
```

训练结果通常保存到：

```text
runs/detect/train
runs/detect/train*
```

最佳权重通常位于：

```text
runs/detect/train/weights/best.pt
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

主要依赖包括：

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

完整依赖以 `requirements.txt` 为准。

## 提交与文件管理注意事项

- 本地推理输出目录 `runs/` 不应作为源码提交。
- `outputs/*.mp4` 和 `outputs/*.avi` 是本地运行产物，不应作为源码提交。
- `assets/test.jpg` 和 `assets/test_video.mp4` 可用于本地验证，但不是项目必须提交的源码。
- 原始 Notebook 和 `Smoke Fire.pt` 是项目复现的重要文件，应保持保留。
