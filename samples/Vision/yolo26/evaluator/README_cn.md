# YOLO26 模型评估 (Evaluation)

本目录包含用于评估 YOLO26 各类任务模型精度的脚本，支持在 RDK 硬件上直接运行并输出标准指标。

## 环境准备

- **Python 环境**: 确保 RDK 已经安装了 Python 3。
- **依赖库**:
  - `pycocotools`: 用于 COCO 数据集 (检测、分割、姿态) 的 mAP 计算。
    ```bash
    pip install pycocotools
    ```
  - `opencv-python`, `numpy` 等基础库。

## 数据集准备

脚本默认从 `datasets/` 目录读取数据，请确保数据集路径正确：
- **检测/分割/姿态**: [COCO val2017](../../../../datasets/coco/README.md)
- **分类**: [ImageNet val](../../../../datasets/imagenet/README.md)


## 输出指标

- **检测/分割/姿态**: 输出 AP @ IoU=0.50:0.95 (all, small, medium, large), AP @ 0.5, AP @ 0.75 以及 Recall 指标。
- **分类**: 输出 Top-1 Accuracy, Top-5 Accuracy 以及推理 FPS。

## 性能测试说明 (Performance Test Instructions)

- **Device列和Model列**: 含义与 Performance Test Instructions 章节的含义相同。

- **计算工具**: 精度数据使用微软官方的无修改的 `pycocotools` 库进行计算。

- **测评模式**:

  - 目标检测 (Object Detection): `iouType="bbox"`

  - 实例分割 (Instance Segmentation): `iouType="bbox"` 和 `iouType="segm"`

  - 人体关键点估计 (Pose Estimation): `iouType="keypoints"`

- **指标含义**:

  - `Accuracy bbox-all mAP @.50:.95` 取自 `Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ]`。

  - `Accuracy bbox-small mAP @.50:.95` 取自 `Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]`。

  - `Accuracy bbox-medium mAP @.50:.95` 取自 `Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]`。

  - `Accuracy bbox-large mAP @.50:.95` 取自 `Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]`。

- **AP vs AR**: AP 更关注“质量”（既要找到目标 Recall，又要框得准且类别对 Precision）；AR 更关注“数量”（只要框住就算，不惩罚误检）。本评估统一取 **AP 指标** 来衡量模型精度。

- **测试数据**: 使用 COCO2017 验证集的 5000 张图片，在板端直接推理，dump 保存为 JSON 文件后送入 `pycocotools` 计算。分数的阈值为 0.25，NMS 的阈值为 0.7。

- **精度差异说明**:

  - `pycocotools` 计算的精度通常比 `ultralytics` 官方工具略低，这是由于 `pycocotools` 取矩形面积而 `ultralytics` 取梯形面积计算 AP 曲线下面积。我们主要关注同一套计算方式下定点模型与浮点模型的对比，以评估量化损失。

  - **分类任务**: 使用 ImageNet-1k 数据集，通过 Top-1 和 Top-5 两个指标来评估。

  - **色彩空间转化**: BPU 模型将 NCHW-RGB888 输入转换为 YUV420SP (NV12) 后会引入细微精度损失。在训练时加入色彩空间转化损失可缓解此问题。

  - **接口差异**: Python 接口和 C/C++ 接口由于在 `memcpy` 转化过程中对浮点数处理方式不同，可能存在细微精度差异。

- **量化说明**: 本表格数据基于 **PTQ (训练后量化)**，使用 50 张图片进行校准。这旨在模拟普通开发者第一次直接编译的精度情况，未进行深度精度调优或 QAT (量化感知训练)，不代表精度的理论上限。