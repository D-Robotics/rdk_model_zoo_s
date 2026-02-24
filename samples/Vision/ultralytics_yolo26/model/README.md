# YOLO26 Models

本目录用于存放转换好的 YOLO26 `.hbm` 模型文件。模型根据目标架构（MARCH）存放在不同的子目录下：

- `nash-e/`: 适用于 RDK S100。
- `nash-m/`: 适用于 RDK S100P。

## 下载说明

我们提供了两个下载脚本，方便您根据需求获取模型：

### 1. 快速下载 (仅 Nano 模型)
适合快速体验和验证功能，仅下载 `yolo26n` 系列模型。
```bash
./download_model.sh
```

### 2. 全量下载 (所有尺寸模型)
下载 `n`, `s`, `m`, `l`, `x` 所有尺寸以及所有任务的模型。
```bash
./download_full_model.sh
```

> **注意**：脚本会自动检测当前硬件环境。如果需要手动下载，可以访问以下地址：
> [https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/YOLO26/](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/YOLO26/)

## 模型列表示例 (以 nash-e 为例)

| 任务 | 模型文件名 (Nano) |
| :--- | :--- |
| 检测 | `nash-e/yolo26n_detect_nashe_640x640_nv12.hbm` |
| 分割 | `nash-e/yolo26n_seg_nashe_640x640_nv12.hbm` |
| 姿态 | `nash-e/yolo26n_pose_nashe_640x640_nv12.hbm` |
| 分类 | `nash-e/yolo26n_cls_nashe_224x224_nv12.hbm` |
| OBB  | `nash-e/yolo26n_obb_nashe_640x640_nv12.hbm` |
