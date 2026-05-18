# Smoke-Fire-Detection-YOLO-V12 Reproduction Guide

This project uses YOLOv12-style Ultralytics workflows for smoke and fire detection. The original notebooks are kept for reference, and the project also provides Python scripts for local Windows inference and training.

## 1. Download the Project

```bash
cd /d D:\AI_Project
git clone https://github.com/the0king0mina/Smoke-Fire-Detection-YOLO-V12.git
cd Smoke-Fire-Detection-YOLO-V12
```

## 2. Create the Conda Environment

```bash
conda create -n yolov12_fire python=3.11 -y
conda activate yolov12_fire
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Check the Environment

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

## 4. Project Structure

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
│   └── README.md
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

## 5. Prepare the Model Weight

The inference scripts use the model weight in the project root:

```text
Smoke Fire.pt
```

Check that it exists:

```bash
dir "Smoke Fire.pt"
```

Do not move it into `assets/` unless you also update `MODEL_PATH` in the scripts.

## 6. Image Detection

Place a test image here:

```text
assets/test.jpg
```

Run:

```bash
python predict_image.py
```

The output is saved by Ultralytics, usually under:

```text
runs/detect/predict
runs/detect/predict2
```

## 7. Video Detection

Place a test video here:

```text
assets/test_video.mp4
```

Run:

```bash
python predict_video.py
```

The output video is saved to:

```text
outputs/fire_output.mp4
```

## 8. Camera Detection

Run:

```bash
python predict_camera.py
```

Press `q` to quit the camera window.

## 9. Prepare a Training Dataset

Prepare a YOLO-format dataset under:

```text
datasets/fire_smoke/
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

Each image should have a matching `.txt` label file with YOLO bounding box format:

```text
class_id x_center y_center width height
```

The default `data.yaml` uses the class order found in the training notebook:

```yaml
path: datasets/fire_smoke
train: images/train
val: images/val
test: images/test

names:
  0: smoke
  1: fire
```

## 10. Train the Model

Run:

```bash
python train_fire.py
```

Default training settings:

```text
initial model: yolo12n.pt
data: datasets/fire_smoke/data.yaml
epochs: 60
imgsz: 640
batch: 16
```

Training results are usually saved under:

```text
runs/detect/train
runs/detect/train2
```

The best weight is usually saved at:

```text
runs/detect/train/weights/best.pt
```

## 11. Common Errors

### Model File Not Found

Example:

```text
FileNotFoundError: Model file not found: Smoke Fire.pt
```

Fix:

1. Confirm that `Smoke Fire.pt` exists in the project root.
2. Do not place the model inside `assets/`.
3. Do not rename the model file unless you also update `MODEL_PATH`.

### Test Image Not Found

Example:

```text
Test image not found: assets/test.jpg
```

Fix:

1. Put a test image in `assets/`.
2. Rename it to `test.jpg`.
3. Run `python predict_image.py` again.

### Test Video Not Found

Example:

```text
Input video not found: assets/test_video.mp4
```

Fix:

1. Put a test video in `assets/`.
2. Rename it to `test_video.mp4`.
3. Run `python predict_video.py` again.

### torch.cuda.is_available() Shows False

This means PyTorch has not detected a CUDA GPU.

Fix:

1. CPU inference can still be used for simple image or video tests.
2. For training, install a PyTorch build that matches your CUDA version.
3. Run `nvidia-smi` to check your NVIDIA driver and GPU status.

### CUDA Out of Memory

Fix:

1. Change `BATCH_SIZE = 16` in `train_fire.py` to `8`.
2. If it still fails, try `4` or `2`.
3. You can also reduce `IMAGE_SIZE = 640` to `512`.
4. Close other programs that are using the GPU.

### Camera Cannot Open

Fix:

1. Check whether the computer has a camera.
2. Check Windows camera permissions.
3. Close other software that may be using the camera.
4. If the default camera is not index `0`, change `CAMERA_INDEX = 0` in `predict_camera.py` to `1`.

## 12. Reproduction Acceptance Checklist

1. `pip install -r requirements.txt` completes successfully.
2. `python predict_image.py` reads `assets/test.jpg` and generates detection output.
3. `python predict_video.py` reads `assets/test_video.mp4` and generates `outputs/fire_output.mp4`.
4. `python predict_camera.py` opens the camera and performs real-time detection.
5. `python train_fire.py` starts training when the dataset exists.
6. All new scripts use relative paths.
7. The original notebooks are still kept.
8. The original `Smoke Fire.pt` file is still kept.
