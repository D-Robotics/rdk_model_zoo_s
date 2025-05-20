English| [简体中文](./README_cn.md)

# Ultralytics YOLO Pose

## Abstract
```bash
D-Robotics OpenExplore Version: >= 3.0.31
Ultralytics YOLO Version: >= 8.3.0
```

## Support Models: 

```bash
- YOLOv8 - Pose
- YOLO11 - Pose
```

## Introduction to YOLO

![](source/imgs/pose-estimation-examples.avif)

YOLO (You Only Look Once) is a popular object detection and image segmentation model developed by Joseph Redmon and Ali Farhadi of the University of Washington. YOLO was introduced in 2015 and quickly gained popularity due to its high speed and accuracy.

 - YOLOv2, released in 2016, improved upon the original model by incorporating batch normalization, anchor boxes, and dimension clustering.
 - YOLOv3: The third iteration of the YOLO model family, originally by Joseph Redmon, known for its efficient real-time object detection capabilities.
 - YOLOv4: A darknet-native update to YOLOv3, released by Alexey Bochkovskiy in 2020.
 - YOLOv5: An improved version of the YOLO architecture by Ultralytics, offering better performance and speed trade-offs compared to previous versions.
 - YOLOv6: Released by Meituan in 2022, and in use in many of the company's autonomous delivery robots.
 - YOLOv7: Updated YOLO models released in 2022 by the authors of YOLOv4.
 - YOLOv8: The latest version of the YOLO family, featuring enhanced capabilities such as instance segmentation, pose/keypoints estimation, and classification.
 - YOLOv9: An experimental model trained on the Ultralytics YOLOv5 codebase implementing Programmable Gradient Information (PGI).
 - YOLOv10: By Tsinghua University, featuring NMS-free training and efficiency-accuracy driven architecture, delivering state-of-the-art performance and latency.
 - YOLO11 🚀: Ultralytics' latest YOLO models delivering state-of-the-art (SOTA) performance across multiple tasks.
 - YOLO12 builds a YOLO framework centered around attention mechanisms, employing innovative methods and architectural improvements to break the dominance of CNN models within the YOLO series. This enables real-time object detection with faster inference speeds and higher detection accuracy.


## Quick Experience

```bash
# Make Sure your are in this file
$ cd samples/Vision/ultralytics_YOLO_Pose

# Check your workspace
$ tree -L 2
.
|-- README.md     # English Document
|-- README_cn.md  # Chinese Document
|-- py
|   `-- ultralitics_YOLO_Pose_YUV420SP.py      # Quick Start
`-- source
    |-- imgs
    |-- reference_hbm_models    # Reference HBM Models
    |-- reference_logs          # Reference logs
    `-- reference_yamls         # Reference yaml configs
```

Run it directly and the model file will be downloaded automatically.

```bash
$ python3 py/ultralitics_YOLO_Pose_YUV420SP.py
```

If you want to replace other models or use other pictures, you can modify the parameters in the script file.

```bash
$ python3 py/ultralitics_YOLO_Pose_YUV420SP.py -h

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Path to BPU Quantized *.bin Model. RDK X3(Module): Bernoulli2. RDK Ultra: Bayes. RDK X5(Module): Bayes-e. RDK S100: Nash-e. RDK S100P: Nash-m.
  --test-img TEST_IMG   Path to Load Test Image.
  --img-save-path IMG_SAVE_PATH
                        Path to Load Test Image.
  --nms-thres NMS_THRES
                        IoU threshold.
  --score-thres SCORE_THRES
                        confidence threshold.
  --reg REG             DFL reg layer.
  --kpt-conf-thres KPT_CONF_THRES
                        confidence threshold.
```


## Result Analysis

![](source/imgs/ultralytics_YOLO_Pose_demo.jpg)

The program automatically downloads the BPU HBM model of YOLO11n-Pose and completes the object detection task of the pictures. The visualization results are saved in the `py_result.jpg` file in the current directory.

## BenchMark - Performance

### RDK S100P

| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv8n-Pose | 640×640 | 80 | 1.8 ms / 527.3 FPS (1 thread  ) <br/> 2.0 ms / 951.9 FPS (2 threads) <br/> 2.7 ms / 1062.4 FPS (3 threads) | 1 ms | 3.3  M | 9.2   B |
| YOLOv8s-Pose | 640×640 | 80 | 2.6 ms / 374.3 FPS (1 thread  ) <br/> 3.4 ms / 577.9 FPS (2 threads)  | 1 ms | 11.6 M | 30.2  B |
| YOLOv8m-Pose | 640×640 | 80 | 4.4 ms / 224.5 FPS (1 thread  ) <br/> 6.9 ms / 285.2 FPS (2 threads)  | 1 ms | 26.4 M | 81.0  B |
| YOLOv8l-Pose | 640×640 | 80 | 8.1 ms / 122.7 FPS (1 thread  ) <br/> 14.3 ms / 138.8 FPS (2 threads) | 1 ms | 44.4 M | 168.6 B |
| YOLOv8x-Pose | 640×640 | 80 | 12.1 ms / 82.4 FPS (1 thread  ) <br/> 22.2 ms / 89.4 FPS (2 threads)  | 1 ms | 69.4 M | 263.2 B |
| YOLO11n-Pose | 640×640 | 80 | 1.8 ms / 524.8 FPS (1 thread  ) <br/> 2.1 ms / 924.0 FPS (2 threads) <br/> 2.9 ms / 1005.0 FPS (3 threads) | 1 ms | 2.9  M | 7.6   B |
| YOLO11s-Pose | 640×640 | 80 | 2.6 ms / 370.9 FPS (1 thread  ) <br/> 3.4 ms / 573.7 FPS (2 threads)  | 1 ms | 9.9  M | 23.2  B |
| YOLO11m-Pose | 640×640 | 80 | 4.8 ms / 204.7 FPS (1 thread  ) <br/> 7.7 ms / 256.5 FPS (2 threads)  | 1 ms | 20.9 M | 71.7  B |
| YOLO11l-Pose | 640×640 | 80 | 5.9 ms / 167.6 FPS (1 thread  ) <br/> 9.9 ms / 199.4 FPS (2 threads)  | 1 ms | 26.2 M | 90.7  B |
| YOLO11x-Pose | 640×640 | 80 | 10.4 ms / 95.3 FPS (1 thread  ) <br/> 18.8 ms / 105.3 FPS (2 threads) | 1 ms | 58.8 M | 203.3 B |

### RDK S100

| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv8n-Pose | 640×640 | 80 | 2.4 ms / 406.9 FPS (1 thread  ) <br/> 2.6 ms / 736.2 FPS (2 threads) <br/> 3.6 ms / 795.1 FPS (3 threads) | 1 ms | 3.3  M | 9.2   B |
| YOLOv8s-Pose | 640×640 | 80 |  3.5 ms / 276.9 FPS (1 thread  ) <br/> 4.7 ms / 411.5 FPS (2 threads) | 1 ms | 11.6 M | 30.2  B |
| YOLOv8m-Pose | 640×640 | 80 |  6.1 ms / 161.0 FPS (1 thread  ) <br/> 9.9 ms / 200.5 FPS (2 threads) | 1 ms | 26.4 M | 81.0  B |
| YOLOv8l-Pose | 640×640 | 80 |  11.3 ms / 88.0 FPS (1 thread  ) <br/> 20.1 ms / 98.8 FPS (2 threads) | 1 ms | 44.4 M | 168.6 B |
| YOLOv8x-Pose | 640×640 | 80 |  17.2 ms / 57.9 FPS (1 thread  ) <br/> 31.8 ms / 62.5 FPS (2 threads) | 1 ms | 69.4 M | 263.2 B |
| YOLO11n-Pose | 640×640 | 80 | 2.4 ms / 395.2 FPS (1 thread  ) <br/> 2.7 ms / 714.3 FPS (2 threads) <br/> 3.9 ms / 749.0 FPS (3 threads) | 1 ms | 2.9  M | 7.6   B |
| YOLO11s-Pose | 640×640 | 80 |  3.5 ms / 276.2 FPS (1 thread  ) <br/> 4.8 ms / 411.1 FPS (2 threads) | 1 ms | 9.9  M | 23.2  B |
| YOLO11m-Pose | 640×640 | 80 | 6.6 ms / 149.7 FPS (1 thread  ) <br/> 10.9 ms / 181.8 FPS (2 threads) | 1 ms | 20.9 M | 71.7  B |
| YOLO11l-Pose | 640×640 | 80 | 8.1 ms / 121.6 FPS (1 thread  ) <br/> 13.8 ms / 143.1 FPS (2 threads) | 1 ms | 26.2 M | 90.7  B |
| YOLO11x-Pose | 640×640 | 80 |  14.5 ms / 68.8 FPS (1 thread  ) <br/> 26.6 ms / 74.8 FPS (2 threads) | 1 ms | 58.8 M | 203.3 B |



### Performance Test Instructions

1. The performance data tested here are all for models with YUV420SP (nv12) input. There is no significant difference in performance data for models with NCHWRGB input.
2. BPU Latency and BPU Throughput.
- Single-thread latency refers to the delay of processing a single frame by a single thread on a single BPU core, representing the ideal scenario for BPU inference task.
- Multi-thread frame rate means multiple threads simultaneously send tasks to the BPU, where each BPU core can handle tasks from multiple threads. Generally, in engineering projects, controlling the frame delay to be relatively small with 4 threads while fully utilizing the BPU up to 100% can achieve a good balance between throughput (FPS) and frame latency. The BPU of S100/S100P performs quite well, usually requiring only 2 threads to fully utilize the BPU, achieving outstanding frame latency and throughput.
- Data recorded in the table typically reaches a point where throughput does not significantly increase with the number of threads.
- BPU latency and BPU throughput were tested on the board using the following commands:
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin

python3 ../../../resource/tools/batch_perf/batch_perf.py --max 3 --file source/reference_hbm_models/
```

3. The test boards were in their optimal state.
- Optimal state for S100P: CPU consists of 6 × A78AE @ 2.0GHz with full-core Performance scheduling, BPU is 1 × Nash-m @ 1.5GHz, delivering 128TOPS at int8.
- Optimal state for S100: CPU consists of 6 × A78AE @ 1.5GHz with full-core Performance scheduling, BPU is 1 × Nash-e @ 1.0GHz, delivering 80TOPS at int8.

```bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"
```


## Advanced Development

### kpt Definition
Ultralytics YOLO Pose's keypoints are based on object detection. The definition of kpt refers to the following:
```python
COCO_keypoint_indexes = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}
```

### Calculation Process Introduction

![](source/imgs/ultralytics_YOLO_Pose_DataFlow.png)

The object detection part of the Ultralytics YOLO Pose model is consistent with Ultralytics YOLO Detect, featuring an additional feature map with Channel = 57 corresponding to 17 Key Points, which include coordinates x, y relative to the downsampled factor of the feature map and the score for that point.

Through the object detection part, once we identify that the Key Points at a certain location meet the requirements, multiplying them by the corresponding downsample factor gives us the Key Points coordinates based on the input size.

### Environment, Project Preparation

Note: Any errors such as "No such file or directory", "No module named 'xxx'", "command not found" should be carefully checked. Do not copy and run commands one by one if you do not understand the modification process; instead, visit the developer community starting from YOLOv5 for better understanding.

- Download the ultralytics/ultralytics repository and set up the environment according to the official documentation of ultralytics.

```bash
git clone https://github.com/ultralytics/ultralytics.git
```

- Navigate into the local repository and download the official pre-trained weights. Here, we use the YOLO11n-Pose model with 2.9 million parameters as an example.

```bash
cd ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
```

### Model Training

- Refer to the official documentation of ultralytics for model training. This document is maintained by ultralytics and is of very high quality. There are also numerous reference materials online, making it not difficult to obtain a pre-trained weight model like the official one.
- Please note, no modifications are needed to any program or the forward method during training.

Official Documentation of Ultralytics YOLO: [https://docs.ultralytics.com/modes/train/](https://docs.ultralytics.com/modes/train/)

### Export to ONNX

- Uninstall the command-line commands related to yolo so that direct modifications to the `./ultralytics/ultralytics` directory can take effect.

```bash
$ conda list | grep ultralytics
$ pip list | grep ultralytics # or
# If exists, uninstall
$ conda uninstall ultralytics 
$ pip uninstall ultralytics   # or
```

If it does not go smoothly, you can confirm the location of the `ultralytics` directory that needs modification using the following Python command.

```bash
>>> import ultralytics
>>> ultralytics.__path__
['/home/wuchao/miniconda3/envs/yolo/lib/python3.11/site-packages/ultralytics']
# or
['/home/wuchao/YOLO11/ultralytics_v11/ultralytics']
```

File Directory: `./ultralytics/ultralytics/nn/modules/head.py`, around line 242, replace the forward method of the `Pose` class with the following content.
Note: It is suggested to keep the original `forward` method, e.g., rename it to `forward_`, for switching back during training.

```python
def forward(self, x):  # RDK
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result

## If the output head order is reversed between bbox and cls, you can modify the append order of cv2 and cv3 as follows,
## then re-export the ONNX and compile it into a hbm model

def forward(self, x):  # RDK
    result = []
    for i in range(self.nl):
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
```

- Other optional optimization modules can be referenced in this repository's README for Ultralytics YOLO Detect.

- Run the following Python script. If there is a **No module named onnxsim** error, install it accordingly.
- Note, if the generated ONNX model shows a too high IR version, set simplify=False. Both settings have no impact on the final bin model but turning it on can improve the readability of the ONNX model in Netron.

```python
from ultralytics import YOLO
YOLO('yolo11n-pose.pt').export(imgsz=640, format='onnx', simplify=False, opset=19)
```

### Prepare Calibration Data

Refer to the minimal calibration data preparation script provided by RDK Model Zoo S: `samples/Vision/ultralytics_YOLO_Detect/source/generate_cal_data.py` for preparing calibration data.

### Model Compilation
```bash
(bpu_docker) $ hb_compile --config config.yaml
```

### Exception Handling

Model Zoo provides compilation logs, bc model information logs, and hbm model logs for comparing your obtained model with the reference models from Model Zoo.

```bash
./samples/Vision/ultralytics_YOLO_Pose/source/reference_logs/
|-- hb_combine_yolo11n_pose.txt
|-- hb_combine_yolov8n_pose.txt
|-- hb_model_info_yolo11n_pose.txt
|-- hb_model_info_yolov8n_pose.txt
|-- hrt_model_exec_model_info_yolo11n_pose.txt
`-- hrt_model_exec_model_info_yolov8n_pose.txt
```

## References

[ultralytics](https://docs.ultralytics.com/)