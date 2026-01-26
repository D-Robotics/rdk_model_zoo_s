# YOLO26 Model Description

This directory describes the complete workflow for using YOLO26 in this Model Zoo, including: algorithm introduction, model conversion, runtime inference (Python), reusable pre/post-processing interface descriptions, and model evaluation steps.

---

## Algorithm Overview



### Algorithm Functions
YOLO26 supports the following tasks:

- **Detection**: Generic object detection (COCO).
- **Segmentation**: Instance segmentation.
- **Pose**: Human keypoint detection.
- **Classification**: Image classification.
- **OBB**: Rotated object detection for aerial scenes.

---

## Directory Structure

This directory contains:

```bash
.
|-- conversion                          # Model conversion workflow
|   `-- README.md                       # Model conversion usage guide
|-- evaluator                           # Model evaluation related content
|   `-- README.md                       # Model evaluation guide
|-- model                               # Model files and download info
|   `-- README.md                       # Model description
|-- runtime                             # Model inference examples
|   `-- python                          # Python inference examples
|       |-- README.md                   # Python inference usage guide
|       |-- README_cn.md                # Python inference usage guide (Chinese)
|       |-- main.py                     # Unified inference entry script
|       |-- main_detect.py              # Detection task entry script
|       |-- main_seg.py                 # Segmentation task entry script
|       |-- main_pose.py                # Pose task entry script
|       |-- main_cls.py                 # Classification task entry script
|       |-- main_obb.py                 # OBB task entry script
|       |-- yolo26_*.py                 # Task-specific wrapper classes
|       `-- ...
`-- README.md                           # YOLO26 sample overall description (this file)
```

---

## QuickStart

To facilitate quick experience, python scripts are provided for immediate execution.

### Python

- Navigate to the `python` directory under `runtime` and execute the specific task script:
    ```bash
    cd runtime/python/
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Run Detection
    python main_detect.py
    ```
- For detailed usage of `python` code, or step-by-step model execution, please refer to `runtime/python/README.md`.

---

## Model Conversion

- The ModelZoo provides pre-adapted HBM model files. The runtime scripts will automatically try to download them if they are missing. If you are not concerned with the model conversion process, **you can skip this section**. Coming soon...

- If you need to customize model conversion parameters or understand the complete model conversion workflow, please refer to `conversion/README.md`.

---

## Runtime

The YOLO26 model inference example is provided in Python to support rapid verification and integration.

### Python Version

    - Provided as scripts, suitable for quick verification of model effects and algorithm flows.
    - Demonstrates the complete process of model loading, inference execution, post-processing, and result visualization.
    - Supports all YOLO26 tasks (Detect/Seg/Pose/Cls/OBB).
    - For specific usage, parameter descriptions, and interface explanations, please refer to `runtime/python/README.md`.

---

## Evaluator

`evaluator/` is used for model accuracy, performance, and numerical consistency evaluation. For detailed instructions, please refer to that directory.

---

## License
Follows the Model Zoo top-level License.
