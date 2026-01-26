# YOLO26 Model Evaluation

This directory contains scripts for evaluating the accuracy of various YOLO26 task models. It supports direct execution on RDK hardware and outputs standard metrics.

## Environment Setup

- **Python Environment**: Ensure Python 3 is installed on the RDK.
- **Dependencies**:
  - `pycocotools`: Used for mAP calculation on COCO datasets (Detection, Segmentation, Pose).
    ```bash
    pip install pycocotools
    ```
  - Base libraries: `opencv-python`, `numpy`, etc.

## Dataset Preparation

Scripts read data from the `datasets/` directory by default. Ensure the paths are correct:
- **Detection / Segmentation / Pose**: [COCO val2017](../../../../datasets/coco/README.md)
- **Classification**: [ImageNet val](../../../../datasets/imagenet/README.md)

## Output Metrics

- **Detection / Segmentation / Pose**: Outputs AP @ IoU=0.50:0.95 (all, small, medium, large), AP @ 0.5, AP @ 0.75, and Recall metrics.
- **Classification**: Outputs Top-1 Accuracy, Top-5 Accuracy, and Inference FPS.

## Performance & Accuracy Evaluation Notes

- **Device and Model Columns**: These carry the same definitions as described in the Performance Test Instructions section.
- **Calculation Tool**: Accuracy data is calculated using the official, unmodified `pycocotools` library from Microsoft.
- **Evaluation Modes**:
  - **Object Detection**: `iouType="bbox"`
  - **Instance Segmentation**: `iouType="bbox"` and `iouType="segm"`
  - **Human Pose Estimation**: `iouType="keypoints"`

- **Metric Definitions**:
  - `Accuracy bbox-all mAP @.50:.95`: Taken from `Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]`.
  - `Accuracy bbox-small/medium/large`: Represents AP for objects of different scales as defined by COCO.

- **AP vs. AR**: AP (Average Precision) focuses on "Quality" (both high Recall and high Precision with accurate localization). AR (Average Recall) focuses on "Quantity" (finding targets regardless of false positives). This evaluation uniformly uses **AP metrics** to measure model accuracy.

- **Test Procedure**: Inference is performed on 5,000 images from the COCO2017 validation set directly on the board. Results are dumped to a JSON file and processed by `pycocotools`.
  - **Score Threshold**: 0.25
  - **NMS Threshold**: 0.7

- **Accuracy Discrepancy Notes**:
  - Metrics calculated by `pycocotools` are typically slightly lower than those from `ultralytics` official tools. This is because `pycocotools` uses rectangular integration for the area under the AP curve, while `ultralytics` uses trapezoidal integration. Our primary focus is the comparison between fixed-point (quantized) and floating-point models using the same methodology to assess quantization loss.
  - **Classification**: Evaluated on the ImageNet-1k dataset using Top-1 and Top-5 accuracy.
  - **Color Space Conversion**: The BPU model introduces minor precision loss when converting NCHW-RGB888 input to YUV420SP (NV12). Incorporating color space conversion loss during training can mitigate this.
  - **Interface Variance**: Minor precision differences may occur between Python and C/C++ interfaces due to different handling of floating-point numbers during `memcpy` and data conversion.

- **Quantization Details**: The data in this report is based on **PTQ (Post-Training Quantization)** using 50 images for calibration. This is intended to simulate the out-of-the-box accuracy a developer would experience upon first compilation. It does not involve deep accuracy tuning or QAT (Quantization-Aware Training) and does not represent the theoretical upper bound of the model's precision.
