from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import CBAM
from ultralytics.yolo.v8.detect.train import DetectionTrainer


class C2fWithCBAM(C2f):
    """在原有 C2f 输出后增加 CBAM。"""

    def forward(self, x):
        x = super().forward(x)
        return self.attention(x)


class CBAMDetectionTrainer(DetectionTrainer):
    """在 YOLOv8s 的 P3 特征融合处增加 CBAM。"""

    def get_model(self, cfg=None, weights=None, verbose=True):
        # 构建原始模型，并加载可以匹配的预训练权重
        model = super().get_model(
            cfg=cfg,
            weights=weights,
            verbose=verbose
        )

        # 当前版本的 YOLOv8s 第 15 层为 P3 分支的 C2f
        layer = model.model[15]

        if not isinstance(layer, C2f):
            raise TypeError(
                '模型第 15 层不是预期的 C2f，请检查模型和 Ultralytics 版本。'
            )

        channels = layer.cv2.conv.out_channels

        # 保留原有 C2f 参数，增加 CBAM
        layer.__class__ = C2fWithCBAM
        layer.attention = CBAM(channels)

        print(f'已在 P3 的 C2f 输出处加入 CBAM，通道数：{channels}')

        return model