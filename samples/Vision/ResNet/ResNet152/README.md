English | [简体中文](./README_cn.md)

# resnet152

- [resnet152](#resnet152)
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

- **Paper**: [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/2307.09283)

- **GitHub repository**: [vision/resnet.py at main · pytorch/vision (github.com)](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

ResNet was proposed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun from Microsoft Research. It won the ILSVRC (ImageNet Large Scale Visual Recognition Challenge) in 2015, achieving a top-5 error rate of 3.57% with fewer parameters than VGGNet. Its outstanding performance marked a milestone in the history of convolutional neural networks. Kaiming He also received the CVPR2016 Best Paper Award for this work.

The main contribution of "ResNet" is the identification of the degradation phenomenon and the use of shortcut connections to address it. This greatly reduces the difficulty of training very deep neural networks, allowing network depth to exceed 100 layers for the first time, with some models even surpassing 1000 layers. The ResNet architecture accelerates training and significantly improves model accuracy. It also generalizes well and can be directly applied to other architectures such as InceptionNet.

ResNet modifies the VGG19 network by adding residual units via shortcut connections. Key changes include using stride-2 convolutions for downsampling and replacing the fully connected layer with a global average pooling layer. An important design principle is that when the feature map size is halved, the number of feature maps is doubled, maintaining network complexity. As shown in the figure, ResNet introduces shortcut connections between every two layers, forming residual learning. The dashed lines indicate changes in the number of feature maps. The 34-layer ResNet shown can be extended to deeper networks. For 18-layer and 34-layer ResNet, residual learning occurs between two layers; for deeper networks, it occurs between three layers with 1x1, 3x3, and 1x1 convolutions. Notably, the number of hidden layer feature maps is relatively small, at 1/4 of the output feature map count.

ResNet152 is a widely used version in the ResNet family, featuring 152 convolutional layers (excluding shortcut connections). Unlike the 18-layer and 34-layer versions, ResNet50 adopts a bottleneck structure for its residual blocks. Each bottleneck block consists of three convolutional layers: a 1x1 convolution for dimensionality reduction, a 3x3 convolution for main feature extraction, and another 1x1 convolution for restoring the original dimensions. This design reduces the number of parameters and computational cost, enabling the construction of deeper networks.

![](./data/ResNet_architecture2.png)
![](./data/ResNet_architecture.png)

**ResNet model features**:

- The residual structure constructs an identity mapping, ensuring that the final error rate does not increase as depth increases.
- ResNet is designed to solve the degradation problem. The residual blocks make it easy to learn identity mappings, so stacking more blocks does not degrade performance.
- The effective depth of the network is determined during training, giving ResNet a degree of deep self-adaptation.
- Deeper networks like ResNet152 use bottleneck residual blocks, which employ 1x1 convolutions for dimensionality reduction and restoration. This design effectively reduces computational cost and the number of parameters, making it possible to build much deeper networks.

## 2. Model Download

**.hbm File Download**:

You can use the [download.sh](./model/download.sh) script to download the .hbm model file for this model structure with one click, making it easy to switch models. Or use the following command line to download:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152_224x224_nv12.hbm
```

**ONNX File Download**:

**ONNX文件**：

The ONNX file can be obtained from the following link: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html

Or download directly with:

```bash
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152.onnx
```

## 3. Deployment Test

After downloading the .hbm file, you can run 'test_resnet152.ipynb' or 's100_inference.py' in the python folder to test the model on the board.

If you need to change the test image, you can download the dataset, put it in the data folder, and modify the image path in the Jupyter notebook or Python script.

![inference](data/image.png)

* Deployment performance test:
```shell
hrt_model_exec perf --model_file ./model/resnet152_224x224_nv12.hbm \
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
- Frame totally latency is: 426.180 ms
- Average latency is: 2.131 ms
- Frame rate is: 463.021 FPS
```
thread_num = 3
```
Running condition:
- Thread number is: 3
- Frame count is: 200
Perf result:
- Frame totally latency is: 1100.839 ms
- Average latency is: 5.504 ms
- Frame rate is: 539.012 FPS
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
hb_compile --model resnet152_large_100.onnx --march nash-e
```

### Model Compilation

After model verification, you can perform quantization compilation using the calibration dataset. A reference yaml file is provided in the yaml folder. Run the following command:

```shell
hb_compile --config yaml/resnet152_config.yaml
```

After compilation, you will find the output in the model_output directory. The file needed for deployment is resnet152_224x224_nv12.hbm.

* Cosine similarity after model quantization:

```shell
 +------------+-------------------+------------------+
 | TensorName | Calibrated Cosine | Quantized Cosine |
 +------------+-------------------+------------------+
 | output     | 0.994397          | 0.992285         |
 +------------+-------------------+------------------+
```

* Toolchain performance reference:

```bash
Summary:
FPS (1 core): 449.03
latency: 2.23 ms
BPU conv original OPs per run: 22,564,831,232
```

### Model Inference

The python directory provides demos for quick inference on both X86 and S100 platforms:
* [x86_inference.py](python/x86_inference.py) supports ONNX, HBIR (.bc), and HBM formats for inference on X86.
* [s100_inference.py](python/s100_inference.py) supports HBM format for inference on the board.

x86_inference.py requires -m and -i to specify the model and image paths, for example:
```shell
python3 python/x86_inference.py -m model_output/resnet152_224x224_nv12_quantized_model.bc -i data/zebra_cls.jpg
```

s100_inference.py requires modifying the model and image paths in the main function.