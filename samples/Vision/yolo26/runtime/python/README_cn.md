# YOLO26 Python 推理示例

本目录提供了 YOLO26 系列模型在 RDK 平台上的统一 Python 推理脚本。通过单一入口 `main.py` 即可支持所有主流任务：目标检测、实例分割、姿态估计、图像分类以及旋转框检测 (OBB)。

---

## 环境准备

1. **硬件**: RDK S100 / S100P
2. **系统**: RDK Ubuntu 系统，需预装 `hbm_runtime`。
3. **Python 库**:
   ```bash
   pip install numpy opencv-python
   ```

---

## 目录结构

```text
python/
├── main.py               # 统一入口脚本 (通过命令行参数切换任务)
├── yolo26_det.py         # 检测模型封装类
├── yolo26_seg.py         # 分割模型封装类
├── yolo26_pose.py        # 姿态模型封装类
├── yolo26_cls.py         # 分类模型封装类
├── yolo26_obb.py         # OBB 模型封装类
├── README.md             # 英文说明文档
└── README_cn.md          # 本文件
```

---

## 使用指南

运行 `main.py` 并使用 `--task` 参数指定任务类型。

**基础语法**:
```bash
python main.py --task <任务名> --model-path <模型路径> --test-img <图片路径> [其他参数]
```

### 1. 目标检测 (`detect`)

```bash
python main.py --task detect \
    --model-path ../../model/yolo26n_detect.hbm \
    --test-img /path/to/image.jpg \
    --score-thres 0.25
```

### 2. 实例分割 (`seg`)

```bash
python main.py --task seg \
    --model-path ../../model/yolo26n_seg.hbm \
    --test-img /path/to/image.jpg
```

### 3. 姿态估计 (`pose`)

```bash
python main.py --task pose \
    --model-path ../../model/yolo26n_pose.hbm \
    --test-img /path/to/person.jpg \
    --kpt-conf-thres 0.5
```

### 4. 图像分类 (`cls`)

```bash
python main.py --task cls \
    --model-path ../../model/yolo26n_cls.hbm \
    --test-img /path/to/animal.jpg \
    --topk 5
```

### 5. 旋转框检测 (`obb`)

```bash
python main.py --task obb \
    --model-path ../../model/yolo26n_obb.hbm \
    --test-img /path/to/aerial.jpg
```

---

## 参数说明

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--task` | **必选**. 任务类型: `detect`, `seg`, `pose`, `cls`, `obb`. | - |
| `--model-path` | **必选**. BPU 量化模型 (`.hbm`) 的路径. | - |
| `--test-img` | 输入测试图片的路径. | `bus.jpg` |
| `--label-file` | 类别标签文件 (`.names`). 默认会根据任务类型自动加载. | `None` |
| `--img-save-path`| 结果可视化图片的保存路径 (分类任务除外). | `result.jpg` |
| `--score-thres` | 置信度阈值 (用于 detect/seg/obb/pose). | `0.25` |
| `--nms-thres` | NMS 的 IoU 阈值. | `0.7` |
| `--topk` | (仅 Cls) 显示前 K 个分类结果. | `5` |
| `--kpt-conf-thres`| (仅 Pose) 关键点可见性阈值. | `0.5` |
| `--angle-sign` | (仅 OBB) 角度解码乘数. | `1.0` |
| `--angle-offset` | (仅 OBB) 角度解码偏移量. | `0.0` |

---

## 注意事项

1.  **模型匹配**: 请确保输入的 `.hbm` 模型与指定的 `--task` 相匹配（例如，不要在 `--task detect` 模式下加载分类模型）。
2.  **辅助模块**: 运行时代码依赖 `../../utils/py_utils` 下的共享工具进行预处理（NV12 转换）、后处理（解码/NMS）以及可视化。