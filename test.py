import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# ==================== 路径设置 ====================

# test.py 所在文件夹作为项目根目录
ROOT = Path(__file__).resolve().parent

# 测试图片和标签目录
IMAGE_DIR = ROOT / 'test' / 'operation interface' / 'images'
LABEL_DIR = ROOT / 'test' / 'operation interface' / 'labels'

# 训练结果目录
RUNS_DIR = ROOT / 'runs' / 'detect'

# 检测统计结果目录
OUTPUT_DIR = ROOT / 'test' / '四组检测统计'

# cbam_model.py 所在目录
CBAM_DIR = ROOT


# ==================== 五组模型权重 ====================

MODELS = {
    'A_100': (
        RUNS_DIR
        / 'A_baseline'
        / 'weights'
        / 'best.pt'
    ),
    'B_CBAM': (
        RUNS_DIR
        / 'B_CBAM'
        / 'weights'
        / 'best.pt'
    ),

    'C_P2': (
        RUNS_DIR
        / 'C_P2'
        / 'weights'
        / 'best.pt'
    ),

    'D_SGD': (
        RUNS_DIR
        / 'D_SGD'
        / 'weights'
        / 'best.pt'
    ),
}


# ==================== 检测参数 ====================

# 置信度阈值
CONF = 0.5

# 预测框与真实框的匹配 IoU 阈值
MATCH_IOU = 0.5

# 模型推理时的 NMS IoU 阈值
NMS_IOU = 0.7

# 输入图像尺寸
IMGSZ = 896

# 单张图片最多保留的预测框数量
MAX_DET = 1000


# ==================== 标签读取 ====================

def load_labels(label_path, image_width, image_height):
    """
    读取 YOLO 格式标签。

    每行格式：
    class x_center y_center width height

    坐标和宽高为相对于图片尺寸的归一化值。
    返回：
    [(类别编号, [x1, y1, x2, y2]), ...]
    """
    labels = []

    lines = label_path.read_text(
        encoding='utf-8-sig'
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f'{label_path.name} 第 {line_number} 行标签格式错误，'
                '每行必须包含类别、中心坐标、宽和高，共 5 项。'
            )

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        coordinates = [
            x_center,
            y_center,
            box_width,
            box_height
        ]

        if not all(np.isfinite(coordinates)):
            raise ValueError(
                f'{label_path.name} 第 {line_number} 行包含无效坐标。'
            )

        if class_id < 0:
            raise ValueError(
                f'{label_path.name} 第 {line_number} 行类别编号小于 0。'
            )

        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < box_width <= 1
            and 0 < box_height <= 1
        ):
            raise ValueError(
                f'{label_path.name} 第 {line_number} 行归一化坐标不合法。'
            )

        # YOLO 中心点格式转换为像素角点格式
        x1 = (x_center - box_width / 2) * image_width
        y1 = (y_center - box_height / 2) * image_height
        x2 = (x_center + box_width / 2) * image_width
        y2 = (y_center + box_height / 2) * image_height

        box = np.array([x1, y1, x2, y2], dtype=float)

        labels.append((class_id, box))

    return labels


# ==================== IoU 计算 ====================

def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比。

    box 格式：[x1, y1, x2, y2]
    """
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])

    intersection_size = np.maximum(
        right_bottom - left_top,
        0
    )

    intersection_area = intersection_size.prod()

    area1 = np.maximum(
        box1[2:] - box1[:2],
        0
    ).prod()

    area2 = np.maximum(
        box2[2:] - box2[:2],
        0
    ).prod()

    union_area = area1 + area2 - intersection_area

    return float(
        intersection_area / max(union_area, 1e-12)
    )


# ==================== 预测框与真实框匹配 ====================

def match_boxes(ground_truth, predictions):
    """
    进行一对一匹配。

    第一步：匹配类别相同、IoU 达到要求的框。
    第二步：对剩余框匹配位置达到要求但类别不同的框。

    每个真实框和预测框最多使用一次。
    """
    used_gt = set()
    used_pred = set()

    correct_matches = []
    wrong_class_matches = []

    # True：匹配同类别
    # False：匹配不同类别
    for same_class in (True, False):
        candidates = []

        for gt_index, (gt_class, gt_box) in enumerate(ground_truth):
            for pred_index, (
                pred_class,
                pred_box,
                confidence
            ) in enumerate(predictions):

                category_matches = (gt_class == pred_class)

                if category_matches != same_class:
                    continue

                overlap = calculate_iou(gt_box, pred_box)

                if overlap >= MATCH_IOU:
                    candidates.append(
                        (
                            overlap,
                            confidence,
                            gt_index,
                            pred_index
                        )
                    )

        # 优先匹配 IoU 较大的框
        candidates.sort(reverse=True)

        for overlap, confidence, gt_index, pred_index in candidates:
            if gt_index in used_gt or pred_index in used_pred:
                continue

            used_gt.add(gt_index)
            used_pred.add(pred_index)

            if same_class:
                correct_matches.append(
                    (gt_index, pred_index, overlap)
                )
            else:
                wrong_class_matches.append(
                    (gt_index, pred_index, overlap)
                )

    # 没有匹配到预测框的真实目标
    missed_count = len(ground_truth) - len(used_gt)

    # 没有匹配到真实目标的预测框
    unmatched_prediction_count = (
        len(predictions) - len(used_pred)
    )

    return (
        correct_matches,
        wrong_class_matches,
        missed_count,
        unmatched_prediction_count
    )


# ==================== CSV 保存 ====================

def save_csv(file_path, rows, field_names):
    with open(
        file_path,
        mode='w',
        newline='',
        encoding='utf-8-sig'
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=field_names
        )

        writer.writeheader()
        writer.writerows(rows)


# ==================== 主程序 ====================

def main():
    # 使 B 组自定义模块可以被导入
    sys.path.insert(0, str(CBAM_DIR))

    cbam_file = CBAM_DIR / 'cbam_model.py'

    if not cbam_file.exists():
        raise FileNotFoundError(
            f'没有找到 B 组所需的 cbam_model.py：\n{cbam_file}'
        )

    import cbam_model

    # 检查权重是否存在
    missing_weights = []

    for model_name, weight_path in MODELS.items():
        if not weight_path.exists():
            missing_weights.append(
                f'{model_name}: {weight_path}'
            )

    if missing_weights:
        raise FileNotFoundError(
            '以下模型权重不存在：\n'
            + '\n'.join(missing_weights)
        )

    # 读取测试图片
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f'图片目录不存在：{IMAGE_DIR}'
        )

    image_files = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp'
        }
    )

    if not image_files:
        raise FileNotFoundError(
            f'图片目录中没有找到图片：{IMAGE_DIR}'
        )

    # 检查是否存在同名标签
    missing_labels = []
    labeled_images = []

    for image_path in image_files:
        label_path = LABEL_DIR / f'{image_path.stem}.txt'

        if label_path.exists():
            labeled_images.append(image_path)
        else:
            missing_labels.append(image_path.name)

    print(f'图片总数：{len(image_files)}')
    print(f'有标签图片：{len(labeled_images)}')
    print(f'缺少标签图片：{len(missing_labels)}')

    if missing_labels:
        print('\n以下图片缺少同名标签，不参与统计：')

        for image_name in missing_labels:
            print('  ', image_name)

    if not labeled_images:
        raise ValueError(
            '没有找到带标签的图片，无法统计检测结果。'
        )

    # 按时间创建结果文件夹，避免覆盖之前结果
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result_dir = OUTPUT_DIR / time.strftime('%Y%m%d_%H%M%S')
    result_dir.mkdir(exist_ok=False)

    all_summary = []

    # 依次检测五组模型
    for model_name, weight_path in MODELS.items():
        print('\n' + '=' * 50)
        print(f'开始评估：{model_name}')
        print(f'权重路径：{weight_path}')
        print('=' * 50)

        model = YOLO(str(weight_path))

        image_rows = []
        totals = Counter()

        for index, image_path in enumerate(
            labeled_images,
            start=1
        ):
            print(
                f'[{model_name}] '
                f'{index}/{len(labeled_images)}：'
                f'{image_path.name}'
            )

            # 模型检测
            result = model.predict(
                source=str(image_path),
                imgsz=IMGSZ,
                conf=CONF,
                iou=NMS_IOU,
                max_det=MAX_DET,
                device=0,
                verbose=False
            )[0]

            image_height, image_width = result.orig_shape

            # 读取真实标注
            label_path = (
                LABEL_DIR / f'{image_path.stem}.txt'
            )

            ground_truth = load_labels(
                label_path,
                image_width,
                image_height
            )

            # 读取模型预测框
            predicted_boxes = (
                result.boxes.xyxy.cpu().numpy()
            )

            predicted_classes = (
                result.boxes.cls.cpu().numpy().astype(int)
            )

            predicted_scores = (
                result.boxes.conf.cpu().numpy()
            )

            predictions = []

            for class_id, box, score in zip(
                predicted_classes,
                predicted_boxes,
                predicted_scores
            ):
                predictions.append(
                    (
                        int(class_id),
                        box,
                        float(score)
                    )
                )

            # 统计匹配结果
            (
                correct_matches,
                wrong_class_matches,
                missed_count,
                unmatched_prediction_count
            ) = match_boxes(
                ground_truth,
                predictions
            )

            row = {
                '图片': image_path.name,
                '真实目标': len(ground_truth),
                '预测框': len(predictions),
                '正确检测': len(correct_matches),
                '类别错误': len(wrong_class_matches),
                '未匹配目标': missed_count,
                '未匹配预测': unmatched_prediction_count,
            }

            image_rows.append(row)

            for key in (
                '真实目标',
                '预测框',
                '正确检测',
                '类别错误',
                '未匹配目标',
                '未匹配预测'
            ):
                totals[key] += row[key]

        # 固定置信度和 IoU 下的精确率、召回率
        if totals['预测框'] > 0:
            precision = (
                totals['正确检测'] / totals['预测框']
            )
        else:
            precision = 0

        if totals['真实目标'] > 0:
            recall = (
                totals['正确检测'] / totals['真实目标']
            )
        else:
            recall = 0

        summary_row = {
            '模型': model_name,
            '统计图片数': len(labeled_images),
            '真实目标': totals['真实目标'],
            '预测框': totals['预测框'],
            '正确检测': totals['正确检测'],
            '类别错误': totals['类别错误'],
            '未匹配目标': totals['未匹配目标'],
            '未匹配预测': totals['未匹配预测'],
            '精确率_固定阈值': round(precision, 6),
            '召回率_固定阈值': round(recall, 6),
        }

        all_summary.append(summary_row)

        # 保存当前模型的逐图统计
        detail_path = (
            result_dir / f'{model_name}_逐图统计.csv'
        )

        save_csv(
            detail_path,
            image_rows,
            [
                '图片',
                '真实目标',
                '预测框',
                '正确检测',
                '类别错误',
                '未匹配目标',
                '未匹配预测',
            ]
        )

        print(f'\n{model_name} 统计完成：')
        print(f'真实目标：{totals["真实目标"]}')
        print(f'预测框：{totals["预测框"]}')
        print(f'正确检测：{totals["正确检测"]}')
        print(f'类别错误：{totals["类别错误"]}')
        print(f'未匹配目标：{totals["未匹配目标"]}')
        print(f'未匹配预测：{totals["未匹配预测"]}')
        print(f'精确率：{precision:.4f}')
        print(f'召回率：{recall:.4f}')

    # 保存五组汇总结果
    summary_path = result_dir / '四组汇总.csv'

    save_csv(
        summary_path,
        all_summary,
        [
            '模型',
            '统计图片数',
            '真实目标',
            '预测框',
            '正确检测',
            '类别错误',
            '未匹配目标',
            '未匹配预测',
            '精确率_固定阈值',
            '召回率_固定阈值',
        ]
    )

    print('\n四组模型检测完成。')
    print(f'结果保存位置：{result_dir}')


if __name__ == '__main__':
    main()