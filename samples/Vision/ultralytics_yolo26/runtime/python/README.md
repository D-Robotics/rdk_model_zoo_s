English | [简体中文](./README_cn.md)

# YOLO26 Python Runtime

This directory contains the unified Python inference script for the YOLO26 model series on the RDK platform. A single entry point (`main.py`) supports all major tasks: Object Detection, Instance Segmentation, Pose Estimation, Image Classification, and Oriented Bounding Box (OBB) detection.

---

## Prerequisites

1. **Hardware**: RDK S100 / S100P
2. **System**: RDK Ubuntu OS with `hbm_runtime` installed (usually pre-installed).
3. **Python Libraries**:
   ```bash
   pip install numpy opencv-python
   ```

---

## Directory Structure

```text
python/
├── main.py               # Unified entry script (multi-task via CLI args)
├── yolo26_det.py         # Detection model wrapper
├── yolo26_seg.py         # Segmentation model wrapper
├── yolo26_pose.py        # Pose model wrapper
├── yolo26_cls.py         # Classification model wrapper
├── yolo26_obb.py         # OBB model wrapper
├── README.md             # This file
└── README_cn.md          # Chinese documentation
```

---


## Usage Guide

Run the `main.py` script and specify the task using the `--task` argument.

**Basic Syntax**:
```bash
python main.py --task <TASK_NAME> --model-path <PATH_TO_HBM> --test-img <PATH_TO_IMAGE> [OPTIONS]
```

### 1. Object Detection (`detect`)

```bash
python main.py --task detect \
    --model-path ../../model/yolo26n_detect.hbm \
    --test-img /path/to/image.jpg \
    --score-thres 0.25
```

### 2. Instance Segmentation (`seg`)

```bash
python main.py --task seg \
    --model-path ../../model/yolo26n_seg.hbm \
    --test-img /path/to/image.jpg
```

### 3. Pose Estimation (`pose`)

```bash
python main.py --task pose \
    --model-path ../../model/yolo26n_pose.hbm \
    --test-img /path/to/person.jpg \
    --kpt-conf-thres 0.5
```

### 4. Image Classification (`cls`)

```bash
python main.py --task cls \
    --model-path ../../model/yolo26n_cls.hbm \
    --test-img /path/to/animal.jpg \
    --topk 5
```

### 5. Oriented Bounding Box (`obb`)

```bash
python main.py --task obb \
    --model-path ../../model/yolo26n_obb.hbm \
    --test-img /path/to/aerial.jpg
```

---

## Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--task` | **Required**. Task type: `detect`, `seg`, `pose`, `cls`, `obb`. | - |
| `--model-path` | **Required**. Path to the BPU quantized model (`.hbm`). | - |
| `--test-img` | Path to the input image. | `bus.jpg` |
| `--label-file` | Path to the label file (`.names`). Defaults are loaded automatically based on task. | `None` |
| `--img-save-path`| Path to save the result visualization (except for `cls`). | `result.jpg` |
| `--score-thres` | Confidence threshold for detection/seg/obb/pose. | `0.25` |
| `--nms-thres` | IoU threshold for NMS. | `0.7` |
| `--topk` | (Cls only) Number of top classes to display. | `5` |
| `--kpt-conf-thres`| (Pose only) Threshold for keypoint visibility. | `0.5` |
| `--angle-sign` | (OBB only) Angle decoding multiplier. | `1.0` |
| `--angle-offset` | (OBB only) Angle decoding offset. | `0.0` |

---

## Notes

1.  **Model Compatibility**: Ensure the input `.hbm` model matches the specified `--task` (e.g., do not load a classification model with `--task detect`).
2.  **Helper Modules**: The runtime relies on shared utilities located in `../../utils/py_utils` for preprocessing (NV12 conversion), postprocessing (decoding/NMS), and visualization.
