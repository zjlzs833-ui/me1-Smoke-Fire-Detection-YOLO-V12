# CODEX_TASK.md

# Smoke-Fire-Detection-YOLO-V12 项目复现任务说明

## 1. 项目目标

请帮助我复现 GitHub 仓库：

```text
https://github.com/the0king0mina/Smoke-Fire-Detection-YOLO-V12.git
```

项目目标是基于 YOLOv12 实现烟雾和火灾检测。请将当前 Notebook 型项目整理成一个可以在 Windows 本地直接运行的 Python 项目，并完成图片检测、视频检测、摄像头检测和训练脚本的整理。

---

## 2. 基本要求

请严格遵守以下要求：

1. 不要删除原始文件。
2. 不要删除 `app.ipynb`。
3. 不要删除 `smoke-fire-detection-yolo-v12.ipynb`。
4. 不要删除 `Smoke Fire.pt`。
5. 所有新增脚本必须使用相对路径。
6. 不允许继续使用作者本机路径，例如：

```text
C:\Users\MINA\Downloads\Smoke Fire.pt
```

7. 所有脚本应兼容 Windows。
8. 所有修改应尽量保持最小化。
9. 如果发现依赖、路径或代码错误，请先说明问题，再修复。
10. 最终需要生成一份可复现说明文档 `REPRODUCE.md`。

---

## 3. 当前仓库情况

该仓库目前主要包含：

```text
Smoke-Fire-Detection-YOLO-V12/
├── README.md
├── requirements.txt
├── Smoke Fire.pt
├── app.ipynb
├── smoke-fire-detection-yolo-v12.ipynb
└── Screenshots/
```

当前项目主要问题：

1. 项目主要是 Jupyter Notebook，不是完整 Python 工程。
2. `app.ipynb` 中可能存在作者电脑的绝对路径。
3. `requirements.txt` 当前格式不规范，多个依赖写在同一行。
4. 项目自带推理权重 `Smoke Fire.pt`。
5. 项目缺少直接运行的 `predict_image.py`、`predict_video.py`、`predict_camera.py` 和 `train_fire.py`。
6. 项目缺少完整本地复现说明。

---

## 4. 请先执行的只读检查任务

在修改文件之前，请先阅读并总结以下文件：

```text
README.md
requirements.txt
app.ipynb
smoke-fire-detection-yolo-v12.ipynb
```

请回答以下问题：

1. 这个仓库是否只有 Notebook，没有标准 Python 脚本？
2. `Smoke Fire.pt` 是否可以作为推理模型权重？
3. `requirements.txt` 是否需要重新整理？
4. `app.ipynb` 中是否存在作者本机绝对路径？
5. 训练 Notebook 中使用的模型是什么？
6. 训练 Notebook 中的 `epochs`、`imgsz` 和数据集路径是什么？
7. 该仓库适合先复现推理还是先重新训练？
8. Windows 本地复现需要哪些步骤？

完成上述只读分析后，再开始修改文件。

---

## 5. 需要修改或新增的文件

请完成以下工程化整理任务。

---

### 5.1 修复 requirements.txt

请将 `requirements.txt` 修改为一行一个依赖，建议内容如下：

```text
ultralytics>=8.2.0
torch>=2.1.0
torchvision
torchaudio
opencv-python
numpy
pandas
matplotlib
Pillow
jupyter
notebook
```

如果你认为某些包版本需要调整，请说明原因。

---

### 5.2 新建 assets 目录

请新建：

```text
assets/
```

并在其中放置一个说明文件：

```text
assets/README.md
```

内容说明：

```markdown
# assets

请将测试图片和测试视频放在此目录下。

默认文件名：

- `test.jpg`：用于图片检测
- `test_video.mp4`：用于视频检测
```

---

### 5.3 新建 outputs 目录

请新建：

```text
outputs/
```

并在其中放置一个说明文件：

```text
outputs/README.md
```

内容说明：

```markdown
# outputs

该目录用于保存模型推理后的输出结果，例如检测后的视频文件。
```

---

### 5.4 新建 predict_image.py

请新建文件：

```text
predict_image.py
```

功能要求：

1. 使用项目根目录下的 `Smoke Fire.pt`。
2. 默认读取 `assets/test.jpg`。
3. 如果模型文件不存在，应给出清晰报错。
4. 如果测试图片不存在，应给出清晰报错。
5. 使用 Ultralytics `YOLO` 进行推理。
6. 推理结果保存到 Ultralytics 默认输出目录。
7. 控制台打印结果保存位置。

建议代码结构如下，可根据项目情况优化：

```python
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("Smoke Fire.pt")
IMAGE_PATH = Path("assets/test.jpg")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"找不到测试图片: {IMAGE_PATH}，请将测试图片放到 assets/test.jpg"
        )

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        save=True
    )

    print("图片检测完成。")
    print("结果通常保存在 runs/detect/predict 或 runs/detect/predict* 目录。")

if __name__ == "__main__":
    main()
```

---

### 5.5 新建 predict_video.py

请新建文件：

```text
predict_video.py
```

功能要求：

1. 使用项目根目录下的 `Smoke Fire.pt`。
2. 默认读取 `assets/test_video.mp4`。
3. 输出视频保存为 `outputs/fire_output.mp4`。
4. 使用 OpenCV 读取视频。
5. 使用 Ultralytics `YOLO` 对每帧进行推理。
6. 将检测框画到视频帧上。
7. 如果模型或视频不存在，应给出清晰报错。
8. 脚本应兼容 Windows。

建议代码结构如下，可根据项目情况优化：

```python
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
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"找不到输入视频: {INPUT_VIDEO}，请将测试视频放到 assets/test_video.mp4"
        )

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

    print("开始视频检测...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

    cap.release()
    writer.release()

    print("视频检测完成。")
    print(f"输出文件: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
```

---

### 5.6 新建 predict_camera.py

请新建文件：

```text
predict_camera.py
```

功能要求：

1. 使用项目根目录下的 `Smoke Fire.pt`。
2. 使用电脑默认摄像头。
3. 实时显示烟雾和火灾检测结果。
4. 按 `q` 退出。
5. 如果摄像头打不开，应给出清晰报错。
6. 脚本应兼容 Windows。

建议代码结构如下，可根据项目情况优化：

```python
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = Path("Smoke Fire.pt")

CONF_THRESHOLD = 0.25
IMAGE_SIZE = 640

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头。请检查摄像头权限或设备连接。")

    print("摄像头检测已启动。按 q 退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        annotated_frame = results[0].plot()
        cv2.imshow("Smoke and Fire Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

### 5.7 新建 train_fire.py

请新建文件：

```text
train_fire.py
```

功能要求：

1. 使用 Ultralytics `YOLO`。
2. 默认使用 `yolo12n.pt` 作为训练初始模型。
3. 默认数据集配置文件为 `datasets/fire_smoke/data.yaml`。
4. 默认训练参数：
   - `epochs=60`
   - `imgsz=640`
   - `batch=16`
5. 如果数据集配置文件不存在，应给出清晰提示。
6. 训练结果保存到 Ultralytics 默认 `runs/detect/train` 目录。

建议代码结构如下，可根据项目情况优化：

```python
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path("datasets/fire_smoke/data.yaml")

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            "找不到数据集配置文件 datasets/fire_smoke/data.yaml。"
            "请先按照 REPRODUCE.md 中的数据集结构准备数据。"
        )

    model = YOLO("yolo12n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=60,
        imgsz=640,
        batch=16,
        verbose=True
    )

    print("训练完成。")
    print("结果通常保存在 runs/detect/train 或 runs/detect/train* 目录。")
    print("最佳权重通常位于 runs/detect/train/weights/best.pt。")

if __name__ == "__main__":
    main()
```

---

### 5.8 新建 datasets/fire_smoke 示例结构

请新建以下目录结构：

```text
datasets/
└── fire_smoke/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
```

请创建 `datasets/fire_smoke/data.yaml`，内容如下：

```yaml
path: datasets/fire_smoke
train: images/train
val: images/val
test: images/test

names:
  0: fire
  1: smoke
```

如果你从 Notebook 中确认类别顺序不是 `fire, smoke`，请按照 Notebook 的真实类别顺序修改。

---

### 5.9 新建 REPRODUCE.md

请新建完整复现说明文件：

```text
REPRODUCE.md
```

内容需要包括：

1. 项目简介。
2. 仓库下载方式。
3. Conda 环境创建命令。
4. 依赖安装命令。
5. 项目文件结构说明。
6. 如何准备测试图片。
7. 如何准备测试视频。
8. 如何运行图片检测。
9. 如何运行视频检测。
10. 如何运行摄像头检测。
11. 如何准备训练数据集。
12. 如何重新训练模型。
13. 结果保存位置。
14. 常见报错与解决方法。
15. 复现验收标准。

`REPRODUCE.md` 中请包含以下命令。

下载项目：

```bash
cd /d D:\AI_Project
git clone https://github.com/the0king0mina/Smoke-Fire-Detection-YOLO-V12.git
cd Smoke-Fire-Detection-YOLO-V12
```

创建环境：

```bash
conda create -n yolov12_fire python=3.11 -y
conda activate yolov12_fire
python -m pip install --upgrade pip
pip install -r requirements.txt
```

检查环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

运行图片检测：

```bash
python predict_image.py
```

运行视频检测：

```bash
python predict_video.py
```

运行摄像头检测：

```bash
python predict_camera.py
```

重新训练：

```bash
python train_fire.py
```

---

## 6. 最终项目结构目标

整理后的项目结构应尽量接近：

```text
Smoke-Fire-Detection-YOLO-V12/
├── README.md
├── REPRODUCE.md
├── requirements.txt
├── Smoke Fire.pt
├── app.ipynb
├── smoke-fire-detection-yolo-v12.ipynb
├── predict_image.py
├── predict_video.py
├── predict_camera.py
├── train_fire.py
├── assets/
│   ├── README.md
│   ├── test.jpg
│   └── test_video.mp4
├── outputs/
│   └── README.md
├── datasets/
│   └── fire_smoke/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
└── Screenshots/
```

---

## 7. 本地执行顺序

请在完成文件整理后，告诉我按照以下顺序执行：

### 7.1 激活环境

```bash
conda activate yolov12_fire
```

### 7.2 安装依赖

```bash
pip install -r requirements.txt
```

### 7.3 检查模型权重是否存在

```bash
dir "Smoke Fire.pt"
```

### 7.4 准备测试图片

将测试图片放到：

```text
assets/test.jpg
```

然后执行：

```bash
python predict_image.py
```

### 7.5 准备测试视频

将测试视频放到：

```text
assets/test_video.mp4
```

然后执行：

```bash
python predict_video.py
```

### 7.6 摄像头检测

```bash
python predict_camera.py
```

### 7.7 重新训练

准备 YOLO 格式数据集后执行：

```bash
python train_fire.py
```

---

## 8. 常见问题处理要求

请在 `REPRODUCE.md` 中加入以下常见问题。

---

### 8.1 找不到 Smoke Fire.pt

错误示例：

```text
FileNotFoundError: 找不到模型文件: Smoke Fire.pt
```

解决方法：

1. 确认项目根目录下是否存在 `Smoke Fire.pt`。
2. 不要把模型放到 `assets/`。
3. 不要修改模型文件名。
4. 如果模型文件名改变，请同步修改脚本中的 `MODEL_PATH`。

---

### 8.2 找不到测试图片

错误示例：

```text
找不到测试图片: assets/test.jpg
```

解决方法：

1. 新建 `assets` 文件夹。
2. 把测试图片命名为 `test.jpg`。
3. 放到 `assets/test.jpg`。

---

### 8.3 找不到测试视频

错误示例：

```text
找不到输入视频: assets/test_video.mp4
```

解决方法：

1. 新建 `assets` 文件夹。
2. 把测试视频命名为 `test_video.mp4`。
3. 放到 `assets/test_video.mp4`。

---

### 8.4 torch.cuda.is_available() 显示 False

说明当前 PyTorch 没有检测到 GPU。

处理方式：

1. 如果只是测试图片或视频推理，可以先用 CPU 跑。
2. 如果要训练，建议重新安装匹配 CUDA 的 PyTorch。
3. 使用 `nvidia-smi` 检查显卡驱动。
4. 根据本机 CUDA 版本安装对应 PyTorch。

---

### 8.5 CUDA out of memory

说明显存不足。

处理方式：

1. 将 batch 从 16 改为 8。
2. 仍然报错则改为 4。
3. 仍然报错则改为 2。
4. 可以将 `imgsz=640` 改为 `imgsz=512`。
5. 关闭其他占用 GPU 的程序。

---

### 8.6 摄像头无法打开

处理方式：

1. 检查电脑是否有摄像头。
2. 检查 Windows 摄像头权限。
3. 关闭占用摄像头的软件。
4. 如果默认摄像头不是 0，可尝试将 `cv2.VideoCapture(0)` 改为 `cv2.VideoCapture(1)`。

---

## 9. 验收标准

请在完成修改后，确保满足以下标准：

1. `pip install -r requirements.txt` 可以正常执行。
2. `python predict_image.py` 可以读取 `assets/test.jpg` 并生成检测结果。
3. `python predict_video.py` 可以读取 `assets/test_video.mp4` 并生成 `outputs/fire_output.mp4`。
4. `python predict_camera.py` 可以打开摄像头并实时检测。
5. `python train_fire.py` 可以在数据集存在时启动训练。
6. 所有新增脚本都使用相对路径。
7. 项目中不再依赖作者电脑的绝对路径。
8. `REPRODUCE.md` 能让一个新用户从零完成复现。
9. 原始 Notebook 文件仍然保留。
10. 原始权重文件 `Smoke Fire.pt` 仍然保留。

---

## 10. 请最终输出给我的内容

完成上述工作后，请向我总结：

1. 你修改了哪些文件？
2. 你新增了哪些文件？
3. 你发现了哪些原项目问题？
4. 这些问题分别如何解决？
5. 我应该按什么顺序运行命令？
6. 如果我要重新训练，需要准备什么数据集？
7. 最终推理结果保存在哪里？
8. 最终训练结果保存在哪里？

请不要只说“完成了”，要给出清晰的操作说明。
