English| [简体中文](./README_cn.md)


# Algorithm Overview

**LaneNet** is an advanced deep learning model designed specifically for real-time lane line detection. Its primary goal is to accurately detect every lane line on the road, even under challenging conditions such as blur, occlusion, or poor lighting.  
This project implements a real-time lane detection deep neural network based on the paper *"Towards End-to-End Lane Detection: an Instance Segmentation Approach"*.  
The network structure mainly includes **ENet / UNet / DeepLabv3+** encoders and decoders, and uses **discriminative loss** for instance segmentation.

LaneNet network architecture:  
![NetWork_Architecture](source/data/source_image/network_architecture.png)

Source code: [lanenet-lane-detection-pytorch](https://github.com/IrohXu/lanenet-lane-detection-pytorch)  
Reference paper: [Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591)

---

# Quick Start

## Download Model

You can download the converted model using the following command:
```bash
wget -P $(dirname $0) https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/lanenet256x512.hbm
```

## Run Inference

1. After downloading the model, modify the model and input image paths in `lanenet_s100_infer.py`.
2. Run the following command to see the output results in the `output` directory:
```bash
python lanenet_s100_infer.py
```

Example results:  
![](source/data/source_image/input.jpg)  
![](source/data/source_image/binary_output.jpg)  
![](source/data/source_image/instance_output.jpg)

---

# Model Quantization

## Environment Setup

Recommended environment:
- python >= 3.6
- torch >= 1.2
- torchvision >= 0.4.0
- numpy >= 1.7
- opencv-python
- pandas
- matplotlib

Install dependencies:
```bash
pip install torch torchvision numpy opencv-python pandas matplotlib
```

## Export ONNX Model

Download the pre-trained model:
```bash
wget -P $(dirname $0) https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/best_model.pth
```

The ONNX export interface is implemented in `test.py`. Run the following command to export the ONNX model:
```bash
python test.py --img ../source/data/source_image/input.jpg --model best_model.pth
```

### Model Quantization

1. Download the quantization dataset -> [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3)  
   Convert it to `.npy` format using the following command. Be sure to modify the `dataset_dir` parameter in the script to the folder where you stored Tusimple:
```bash
python get_calibration_data.py
```

2. Run the OE toolchain Docker, mount the directory to the LaneNet development folder, and then run the following command in the Docker terminal to start model conversion:
```bash
hb_compile -c source/yaml/config.yaml
```

---

# Performance Evaluation

## Speed Test

Run the following command on the S100 platform to test performance:
```bash
hrt_model_exec perf --model_file lanenet256x512.hbm 
```

Example output:
```
root@ubuntu:~/lanenet# hrt_model_exec perf --model_file lanenet256x512.hbm 
[UCP]: log level = 3
[UCP]: UCP version = 3.3.3
[VP]: log level = 3
[DNN]: log level = 3
[HPL]: log level = 3
[UCPT]: log level = 6
[DSP]: log level = 3
hrt_model_exec perf --model_file lanenet256x512.hbm

 [Warning]: These operators have range limitations on input data: 
 [Acos, Acosh, Asin, Atanh, BevPoolV2, Div, Gather, GatherElements, GatherND, GridSample, ImageDecoder, IndexSelect, Log, Mod, Pow, Reciprocal, RoiAlign, ScatterElements, ScatterND, Slice, Sqrt, Tan, Tile, Topk, Upsample]. 
 Please make sure that these operators are not in your model, when no input data is provided to the tool. 
 [Suggestion]: Using --input_file command to specify perf input data, which can appoint valid input data.  

[BPU][[BPU_MONITOR]][281473344962784][INFO]BPULib verison(2, 1, 2)[0d3f195]!
[DNN] HBTL_EXT_DNN log level:6
[DNN]: 3.3.3_(4.1.17 HBRT)
Load model to DDR cost 475.583ms.
Frame count: 200,  Thread Average: 14.245405 ms,  thread max latency: 38.879002 ms,  thread min latency: 13.900000 ms,  FPS: 69.897171

Running condition:
  Thread number is: 1
  Frame count   is: 200
  Program run time: 2861.473 ms
Perf result:
  Frame totally latency is: 2849.081 ms
  Average    latency    is: 14.245 ms
  Frame      rate       is: 69.894 FPS
```

## Accuracy Test

After quantization, the cosine similarity results are shown below. All three outputs maintain high similarity:
![](source/data/source_image/result.jpg)
