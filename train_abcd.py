import cv2
from ultralytics import YOLO
from cbam_model import CBAMDetectionTrainer
from pathlib import Path

GROUP = 'A'

BASE_DIR = Path(__file__).resolve().parent

WEIGHTS = str(BASE_DIR / 'yolov8s.pt')
DATA = str(BASE_DIR / 'data.yaml')
PROJECT = str(BASE_DIR / 'runs' / 'detect')

def common_args():
    return dict(
        data=DATA,
        epochs=100,
        patience=0,
        imgsz=896,
        batch=4,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        box=7.5,
        cls=0.5,
        close_mosaic=10,
        label_smoothing=0.1,
        mosaic=1.0,
        fliplr=0.5,
        seed=0,
        device=0,
        plots=True,
        save=True,
        project=PROJECT,
        exist_ok=False
    )

def train_a():
    model = YOLO(WEIGHTS)
    model.train(
        name='A_baseline',
        optimizer='auto',
        **common_args()
    )

def train_b():
    args = dict(
        model=WEIGHTS,
        name='B_CBAM',
        optimizer='auto',
        **common_args()
    )
    trainer = CBAMDetectionTrainer(overrides=args)
    trainer.train()

def train_c():
    model = YOLO('yolov8s-p2.yaml')
    model.load(WEIGHTS)
    model.train(
        name='C_P2',
        optimizer='auto',
        **common_args()
    )

def train_d():
    model = YOLO(WEIGHTS)
    model.train(
        name='D_SGD',
        optimizer='SGD',
        **common_args()
    )

if __name__ == '__main__':
    if GROUP == 'A':
        train_a()
    elif GROUP == 'B':
        train_b()
    elif GROUP == 'C':
        train_c()
    elif GROUP == 'D':
        train_d()