# 基于 YOLOv8s 的火电厂监控界面操作热点检测

本项目使用 YOLOv8s 检测火电厂 DCS 监控界面中的操作热点。针对界面中目标尺寸小、排列密集的问题，采用重叠切片处理图像，并对不同模型结构和训练设置进行比较。

## 项目文件

```text
├── data.yaml            数据集配置文件
├── train_abcd.py        四组实验训练代码
├── cbam_model.py        CBAM 相关代码
├── baseline.py          基线训练代码
├── experiment_C.py      数据增强参数实验代码
├── test.py              检测代码
├── yolov8s.pt           YOLOv8s 预训练权重
└── datasets             数据集
    ├── images
    │   ├── train
    │   └── val
    └── labels
        ├── train
        └── val
```

## 实验环境

- Windows
- Python 3.9
- PyTorch 2.5.1
- Ultralytics 8.0.120
- NVIDIA GeForce RTX 3050 Laptop GPU

## 实验方案

| 组别 | 实验方案 |
|---|---|
| A | YOLOv8s baseline，自动优化器设置 |
| B | YOLOv8s + CBAM 注意力模块 |
| C | YOLOv8s-P2 |
| D | YOLOv8s，使用 SGD 优化器 |

## 训练方法

运行前检查 `data.yaml` 中的数据集路径，以及 `train_abcd.py` 中的预训练权重和结果保存路径。本仓库中的脚本默认以脚本所在文件夹为项目根目录。

在 `train_abcd.py` 中修改 `GROUP`，选择要训练的组别：

```python
GROUP = 'A'
```

运行训练代码：

```bash
python train_abcd.py
```

将 `GROUP` 分别改为 `B`、`C`、`D`，可运行其他组的训练。训练结果保存在代码中设置的 `runs/detect` 目录下。


## 实验结果

下表为各组验证集 mAP50–95 最高轮次的结果。

| 组别 | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| A | 0.85258 | 0.80590 | 0.85617 | 0.62954 |
| B | 0.85586 | 0.80488 | 0.83304 | 0.62078 |
| C | 0.78965 | 0.66963 | 0.79107 | 0.53613 |
| D | 0.84088 | 0.79975 | 0.86107 | 0.66633 |

D 组的 mAP50–95 最高，较 A 组提高 0.03679。B 组和 C 组的 mAP50–95 均低于 A 组。上述结果来自训练期间的验证集。

## 模型权重

各组训练得到的模型权重已发布，可在
[Release 页面](https://github.com/doubleup77777/YOLOv8s_hotspot_detection/releases)
下载。

B 组权重加载时需要使用仓库中的 `cbam_model.py`。


## 论文

《基于 YOLOv8s 的火电厂监控界面操作热点检测方法优化》
