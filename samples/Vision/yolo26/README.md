English | [简体中文](./README_cn.md)

# YOLO26 Model Description

This directory describes the complete workflow for using YOLO26 in this Model Zoo, including: algorithm introduction, model conversion, runtime inference (Python), reusable pre/post-processing interface descriptions, and model evaluation steps.

---

## Algorithm Overview

Ultralytics YOLO26 is the latest evolution of the YOLO series of real-time object detectors, designed from the ground up for edge and low-power devices. It introduces a simplified design that eliminates unnecessary complexity while integrating targeted innovations for faster, lighter, and more accessible deployment.

The architecture of YOLO26 follows three core principles:

- **Simplicity**: YOLO26 is a native end-to-end model that generates prediction results directly without Non-Maximum Suppression (NMS). By eliminating this post-processing step, inference becomes faster, lighter, and easier to deploy in real-world systems.
- **Deployment Efficiency**: The end-to-end design eliminates an entire stage of the pipeline, greatly simplifying integration, reducing latency, and making deployment more robust in various environments.
- **Training Innovation**: Introduces the MuSGD optimizer, a hybrid of SGD and Muon, which brings enhanced stability and faster convergence.

### Algorithm Functions
YOLO26 supports the following tasks:

- **Detection**: Generic object detection (COCO).
- **Segmentation**: Instance segmentation.
- **Pose**: Human keypoint detection.
- **Classification**: Image classification.
- **OBB**: Rotated object detection.

### Original Material
The official resources for YOLO26 are as follows:
- YOLO26 Repo: https://github.com/ultralytics/ultralytics (YOLO26 is integrated into the latest Ultralytics framework)

---

## Directory Structure

This directory contains:

```bash
.
|-- conversion                          # Model conversion workflow
|   |-- onnx_export                     # ONNX export related code
|   |-- mapper.py                       # Model quantization script (ONNX -> HBM)
|   |-- README.md                       # Model conversion guide (English)
|   `-- README_cn.md                    # Model conversion guide (Chinese)
|-- evaluator                           # Model evaluation related content
|   |-- eval_yolo26_*.py                # Task-specific evaluation scripts
|   |-- README.md                       # Model evaluation guide (English)
|   `-- README_cn.md                    # Model evaluation guide (Chinese)
|-- model                               # Model files and download info
|   |-- download_model.sh               # HBM model download script
|   `-- README.md                       # Model download guide
|-- runtime                             # Model inference examples
|   `-- python                          # Python inference examples
|       |-- main.py                     # Unified inference entry script
|       |-- run.sh                      # One-click execution script
|       |-- yolo26_*.py                 # Task-specific wrapper classes
|       |-- README.md                   # Python inference guide (English)
|       `-- README_cn.md                # Python inference guide (Chinese)
|-- test_data                           # Inference results and sample data (empty)
|-- README.md                           # YOLO26 overall description (this file)
`-- README_cn.md                        # YOLO26 overall description (Chinese)
```

---

## QuickStart

To facilitate quick experience, a `run.sh` script and a unified entry script are provided:
- Check system environment and install necessary dependencies;
- Automatically handle model download logic (if the path is missing);
- Run the corresponding Python script for inference verification.

### Python

- Navigate to the `runtime/python/` directory and run the `run.sh` script for a quick experience:
    ```bash
    cd runtime/python/
    ./run.sh
    ```
- For detailed usage of `python` code, or step-by-step model execution, please refer to [runtime/python/README.md](./runtime/python/README.md).

---

## Model Conversion

- The ModelZoo provides pre-adapted HBM model files. The runtime scripts will automatically try to download them if they are missing. If you are not concerned with the model conversion process, **you can skip this section**.

- If you need to customize model conversion parameters or understand the complete model conversion workflow, please refer to [conversion/README.md](./conversion/README.md).

---

## Runtime

The YOLO26 model inference example is provided in Python to support rapid verification and integration.

### Python Version

- Provided as scripts, suitable for quick verification of model effects and algorithm flows;
- Demonstrates the complete process of model loading, inference execution, post-processing, and result visualization;
- Supports all YOLO26 tasks (Detect/Seg/Pose/Cls/OBB);
- For specific usage, parameter descriptions, and interface explanations, please refer to [runtime/python/README.md](./runtime/python/README.md);

---

## Evaluator

`evaluator/` is used for model accuracy, performance, and numerical consistency evaluation. For detailed instructions, please refer to [evaluator/README.md](./evaluator/README.md).

---

## License
Follows the Model Zoo top-level License.