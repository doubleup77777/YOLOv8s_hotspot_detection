import os
import cv2
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def train_model():
    # 加载 YOLOv8s 预训练权重
    model = YOLO(str(BASE_DIR / 'yolov8s.pt'))

    results = model.train(
        data=str(BASE_DIR / 'data.yaml'),
        epochs=100,
        imgsz=896,
        batch=4,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        optimizer='auto',
        box=7.5,
        cls=0.5,
        close_mosaic=10,
        label_smoothing=0.1,
        project=str(BASE_DIR / 'runs' / 'detect'),
        name='dcs_hotspot',
        device=0,
        plots=True,
        save=True
    )


if __name__ == '__main__':
    train_model()