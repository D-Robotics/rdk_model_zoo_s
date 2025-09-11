English | [简体中文](./README_cn.md)

# MobileNetV4

- [MobileNetV4](#mobilenetv4)
  - [1. Introduction](#1-introduction)
  - [2. Model Download](#2-model-download)
  - [3. Deployment Test](#3-deployment-test)
  - [4. Quantization Experiments](#4-quantization-experiments)
    - [Dataset Preparation](#dataset-preparation)
    - [Calibration Data Processing](#calibration-data-processing)
    - [Model Verification](#model-verification)
    - [Model Compilation](#model-compilation)
    - [Model Inference](#model-inference)

## 1. Introduction

- **Paper**: [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)

- **GitHub repository**: [pytorch-image-models/timm/models/MobileNetV4.py at main · huggingface/pytorch-image-models (github.com)](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py)

![](./data/MobileNetV4_architecture.png)

MobileNetV4 is based on classic components of MobileNet, such as separable deep convolution (DW) and point-by-point (PW) expansion and projection inverted bottleneck block, introducing the **Universal Inverted Neck Structure** (UIB). This structure is quite simple, introducing two optional deep convolutions (DW) in the inverted bottleneck block, one located before the extension layer and the other located between the extension layer and the projection layer. The existence or non-existence of these two deep convolutions is part of the neural architecture search (NAS) optimization process, which ultimately generates novel network architectures. Although this modification seems simple, the author cleverly unifies several important modules, including the original inverted neck structure, ConvNext, and FFN in ViT. In addition, UIB has introduced a new variant: ExtraDW. With the enhancement of this technology, the MNv4-Hybrid-Large model achieved an accuracy of 87% on ImageNet-1K and a running time of only 3.8ms on Pixel 8 EdgeTPU.

**MobileNetV4 model features**:

- Introduced the Universal Inverted Bottleneck (UIB) search block, a unified and flexible structure that combines Inverted Bottleneck (IB), ConvNext, Feedforward Networks (FFN), and a new Extra Depth convolution (Extra Depthwise) variant.
- A mobile version of Multi-Head Attention (Mobile MQA) optimized for mobile accelerators is proposed, which provides 39% inference acceleration compared to traditional Multi-Head Self-Attention (MHSA). An optimized neural architecture search (NAS) method is introduced, which improves the effectiveness of MNv4 search.

## 2. Model Download

**.hbm File Download**:

You can use the [download.sh](./model/download.sh) script to download the .hbm model file for this model structure with one click, making it easy to switch models. Or use the following command line to download:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv4_medium_256x256_nv12.hbm
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv4_small_224x224_nv12.hbm
```

**ONNX File Download**:

The ONNX model is converted from the timm library (PyTorch Image Models). Install the required packages with:

```shell
pip install timm onnx
```

After installing the necessary libraries, you can use the [get_mobilenetv4_onnx.py](python/get_mobilenetv4_onnx.py) script in the python folder to download the ONNX file.

* Note: You need to configure a terminal proxy and log in to Huggingface using the following command:
```shell
huggingface-cli login
```

* If you do not want to configure a terminal proxy, you can manually download the model from [timm/mobilenetv4_medium.e500_r256_in1k](https://huggingface.co/timm/mobilenetv4_medium.e500_r256_in1k) and [timm/mobilenetv4_small.e2400_r224_in1k](https://huggingface.co/timm/mobilenetv4_small.e2400_r224_in1k) and use the [python/timm2onnx.py](python/timm2onnx_local.py) script to convert to ONNX.
    
After exporting the ONNX file, the script will output model input, mean, std, path, parameters, etc., in the following format:

```bash
Processing mobilenetv4_small...
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv4_small.onnx
Total number of parameters in the model: 3761480

Processing mobilenetv4_medium...
input: (3, 256, 256)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv4_medium.onnx
Total number of parameters in the model: 9681560
```

## 3. Deployment Test

After downloading the .hbm file, you can run 'test_mobilenetv4.ipynb' or 's100_inference.py' in the python folder to test the model on the board.

If you need to change the test image, you can download the dataset, put it in the data folder, and modify the image path in the Jupyter notebook or Python script.

![inference](data/image.png)

* Deployment performance test:
```shell
hrt_model_exec perf --model_file ./model/mobilenetv4_224x224_nv12.hbm \
                                     --core_id=0 \
                                     --frame_count=200 \
                                     --perf_time=0 \
                                     --thread_num=1
```

mobilenetv4_medium：

thread_num = 1
```
Running condition:
- Thread number is: 1
- Frame count is: 200
Perf result:
- Frame totally latency is: 118.902 ms
- Average latency is: 0.595 ms
- Frame rate is: 1612.084 FPS
```
thread_num = 3
```
Running condition:
- Thread number is: 3
- Frame count is: 200
Perf result:
- Frame totally latency is: 192.371 ms
- Average latency is: 0.962 ms
- Frame rate is: 3000.570 FPS
```

mobilenetv4_small：

thread_num = 1
```
Running condition:
- Thread number is: 1
- Frame count is: 200
Perf result:
- Frame totally latency is: 77.684 ms
- Average latency is: 0.388 ms
- Frame rate is: 2423.449 FPS
```
thread_num = 3
```
Running condition:
- Thread number is: 3
- Frame count is: 200
Perf result:
- Frame totally latency is: 111.443 ms
- Average latency is: 0.557 ms
- Frame rate is: 5130.441 FPS
```


## 4. Quantization Experiments

### Dataset Preparation

The model uses the [ImageNet](https://image-net.org/) dataset.
* Dataset: ILSVRC2012

| Dataset Name | Number of Classes | Number of Images |
| -- | -- | -- |
| ILSVRC2012 Training Set | 1000 classes | ~1.2 million images |
| ILSVRC2012 Validation Set | 1000 classes | 50,000 images |
| ILSVRC2012 Test Set | 1000 classes | 100,000 images |

It is recommended to extract the downloaded dataset into the following structure:

```shell
imagenet/ 
 ├── calibration_data 
 │   ├── ILSVRC2012_val_00000001.JPEG 
 │   ├── ... 
 │   └── ILSVRC2012_val_00000100.JPEG 
 ├── ILSVRC2017_val.txt 
 ├── val 
 │   ├── ILSVRC2012_val_00000001.JPEG 
 │   ├── ... 
 │   └── ILSVRC2012_val_00050000.JPEG 
 └── val.txt
```

### Calibration Data Processing

After preparing 100 calibration images, run the following command to generate calibration data, which will be saved in the /calibration_data_rgb directory:

* Note: "Medium" and "Small" correspond to different image sizes. Please modify the parameters in the main function as needed.

```shell
python3 python/get_calibration_data.py
```

### Model Verification

After preparing the ONNX model, you can quickly verify the model with the following command:

```shell
hb_compile --model mobilenetv4_medium.onnx --march nash-e
```

```shell
hb_compile --model mobilenetv4_small.onnx --march nash-e
```

### Model Compilation

After model verification, you can perform quantization compilation using the calibration dataset. A reference yaml file is provided in the yaml folder. Run the following command:

```shell
hb_compile --config yaml/mobilenetv4_medium_config.yaml
```

```shell
hb_compile --config yaml/mobilenetv4_small_config.yaml
```

After compilation, you will find the output in the model_output directory. The file needed for deployment mobilenetv4_medium_256x256_nv12.hbm and mobilenetv4_medium_224x224_nv12.hbm 

* Cosine similarity after model quantization:

mobilenetv4_medium：

```shell
 +------------+-------------------+------------------+
 | TensorName | Calibrated Cosine | Quantized Cosine |
 +------------+-------------------+------------------+
 | output     | 0.999759          | 0.999863         |
 +------------+-------------------+------------------+
```

mobilenetv4_small:

```shell
 +------------+-------------------+------------------+
 | TensorName | Calibrated Cosine | Quantized Cosine |
 +------------+-------------------+------------------+
 | output     | 0.999892          | 0.99988          |
 +------------+-------------------+------------------+
```

* Toolchain performance reference:

mobilenetv4_medium：

```bash
Summary:
FPS (1 core): 2468.07
latency: 0.41 ms (405.2 us)
BPU conv original OPs per run: 2,160,488,448
```

mobilenetv4_small:

```bash
Summary:
FPS (1 core): 5698.18
latency: 0.18 ms (175.5 us)
BPU conv original OPs per run: 372,011,136
```

### Model Inference

The python directory provides demos for quick inference on both X86 and S100 platforms:
* [x86_inference.py](python/x86_inference.py) supports ONNX, HBIR (.bc), and HBM formats for inference on X86.
* [s100_inference.py](python/s100_inference.py) supports HBM format for inference on the board. using the new HB_HBMRuntime API

x86_inference.py requires -m and -i to specify the model and image paths, for example:
```shell
python3 python/x86_inference.py -m model_output/mobilenetv4_224x224_nv12_quantized_model.bc -i data/zebra_cls.jpg
```
