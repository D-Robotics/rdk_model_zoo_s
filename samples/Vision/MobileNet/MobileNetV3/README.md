English | [简体中文](./README_cn.md)

# mobilenetv3

- [mobilenetv3](#mobilenetv3)
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

- **Paper**: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

- **GitHub repository**: [pytorch-image-models/timm/models/mobilenetv3.py at main · huggingface/pytorch-image-models (github.com)](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py)

![](./data/MobileNetV3_architecture.png)

MobileNetV3 is an improvement on [MobileNetV2](../MobileNetV2/README.md) , which is also a lightweight neural network. MobileNetV3 optimizes the network by using **network architecture search (NAS)** and NetAdapt. At the same time, the paper proposes two different networks, MobileNetV3-Large and MobileNetV3-Small, to cope with different practical use cases. Compared with [MobileNetV2](../MobileNetV2/README.md) MobileNetV3-Large has a 3.2% higher accuracy in ImageNet dataset recognition, while reducing latency by 15%, while MobileNetV3-Small has a 4.6% higher accuracy and reduces latency by 5%. The detection accuracy of MobileNetV3-Large in MS COCO dataset is roughly 25% faster than [MobileNetV2](../MobileNetV2/README.md).

**MobileNetV3 model features**:

- **Depthwise Separable Convolutions**: MobileNetV3 retains the depthwise separable convolutions from MobileNetV2, splitting standard convolutions into depthwise and pointwise convolutions to significantly reduce computation and parameter count
- **Inverted Residuals**: Similar to MobileNetV2, MobileNetV3 uses inverted residual blocks, which consist of expansion, depthwise, and pointwise convolutions, with skip connections between the input and output
- **Squeeze-and-Excitation (SE) Modules**: MobileNetV3 introduces SE modules, which use global pooling and channel attention mechanisms to recalibrate channel weights, enhancing feature representation.
- **H-Swish Activation Function**: MobileNetV3 employs the H-Swish (Hard-Swish) activation function, a computationally efficient variant of the Swish activation, providing a good balance between accuracy and efficiency.
- **Neural Architecture Search (NAS)**: Parts of MobileNetV3's architecture were optimized using NAS, which automates the design process to find the best trade-off between performance and efficiency under various device constraints.

## 2. Model Download

**.hbm File Download**:

You can use the [download.sh](./model/download.sh) script to download the .hbm model file for this model structure with one click, making it easy to switch models. Or use the following command line to download:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv3_224x224_nv12.hbm
```

**ONNX File Download**:

The ONNX model is converted from the timm library (PyTorch Image Models). Install the required packages with:

```shell
pip install timm onnx
```

After installing the necessary libraries, you can use the [get_mobilenetv3_onnx.py](python/get_mobilenetv3_onnx.py) script in the python folder to download the ONNX file.

* Note: You need to configure a terminal proxy and log in to Huggingface using the following command:
```shell
huggingface-cli login
```

* If you do not want to configure a terminal proxy, you can manually download the model from [timm/mobilenetv3_large_100.ra_in1k](https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k) and use the [python/timm2onnx.py](python/timm2onnx_local.py) script to convert to ONNX.
    
After exporting the ONNX file, the script will output model input, mean, std, path, parameters, etc., in the following format:

```shell
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv3_large_100.onnx
Total number of parameters in the model: 5470832
```

## 3. Deployment Test

After downloading the .hbm file, you can run 'test_mobilenetv3.ipynb' or 's100_inference.py' in the python folder to test the model on the board.

If you need to change the test image, you can download the dataset, put it in the data folder, and modify the image path in the Jupyter notebook or Python script.

![inference](data/image.png)

* Deployment performance test:
```shell
hrt_model_exec perf --model_file ./model/mobilenetv3_224x224_nv12.hbm \
                                     --core_id=0 \
                                     --frame_count=200 \
                                     --perf_time=0 \
                                     --thread_num=1
```
thread_num = 1
```
Running condition:
- Thread number is: 1
- Frame count is: 200
Perf result:
- Frame totally latency is: 108.508 ms
- Average latency is: 0.543 ms
- Frame rate is: 1771.793 FPS
```
thread_num = 3
```
Running condition:
- Thread number is: 3
- Frame count is: 200
Perf result:
- Frame totally latency is: 176.123 ms
- Average latency is: 0.881 ms
- Frame rate is: 3276.057 FPS
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

```shell
python3 python/get_calibration_data.py
```

### Model Verification

After preparing the ONNX model, you can quickly verify the model with the following command:

```shell
hb_compile --model mobilenetv3_large_100.onnx --march nash-e
```

### Model Compilation

After model verification, you can perform quantization compilation using the calibration dataset. A reference yaml file is provided in the yaml folder. Run the following command:

```shell
hb_compile --config yaml/mobilenetv3_config.yaml
```

After compilation, you will find the output in the model_output directory. The file needed for deployment is mobilenetv3_224x224_nv12.hbm.

* Cosine similarity after model quantization:

```shell
 +------------+-------------------+------------------+
 | TensorName | Calibrated Cosine | Quantized Cosine |
 +------------+-------------------+------------------+
 | output     | 0.911233          | 0.909042         |
 +------------+-------------------+------------------+
```

* Toolchain performance reference:

```bash
Summary:
FPS (1 core): 2616.81
latency: 0.38 ms (382.1 us)
BPU conv original OPs per run: 433,179,520
```

### Model Inference

The python directory provides demos for quick inference on both X86 and S100 platforms:
* [x86_inference.py](python/x86_inference.py) supports ONNX, HBIR (.bc), and HBM formats for inference on X86.
* [s100_inference.py](python/s100_inference.py) supports HBM format for inference on the board. using the new HB_HBMRuntime API

x86_inference.py requires -m and -i to specify the model and image paths, for example:
```shell
python3 python/x86_inference.py -m model_output/mobilenetv3_224x224_nv12_quantized_model.bc -i data/zebra_cls.jpg
```
