English | [简体中文](./README_cn.md)

# MobileNetV1

- [MobileNetV1](#mobilenetv1)
  - [1. Introduction](#1-introduction)
  - [2. Model Download](#2-model-download)
  - [3. Deployment and Testing](#3-deployment-and-testing)

## 1. Introduction

- **Paper**: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

- **GitHub Repository**: [models/research/slim/nets/mobilenet_v1.md at master · tensorflow/models (github.com)](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

MobileNetV1 introduces a lightweight neural network designed for embedded devices. It utilizes depthwise separable convolutions to construct efficient deep neural networks. The core idea is to decompose standard convolutions into **depthwise convolutions** and **pointwise convolutions**. By separating standard convolutions, the number of intermediate feature maps is reduced, effectively decreasing the number of network parameters.

![](./data/depthwise&pointwise.png)

**Key Features of MobileNetV1:**

- **Depthwise Separable Convolutions**: MobileNet is based on depthwise separable convolutions, a form of **factorized convolution** that splits a standard convolution into a depthwise convolution and a 1×1 convolution called pointwise convolution, first introduced in InceptionV3.
- **Hyperparameters**: The width multiplier $\alpha$ and resolution multiplier $\rho$ are used to reduce computational cost and model size.

## 2. Model Download

**.hbm File Download:**

Go to the model folder and use the following command to download the MobileNetV1 model:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/mobilenetv1_224x224_nv12.hbm
```

This model is the quantized output produced by Horizon's reference algorithm.

Model conversion uses the Caffe model: https://github.com/shicai/MobileNet-Caffe

For quantization and conversion steps of MobileNetV1, refer to the conversion steps of other MobileNet models or directly use the samples in the OE development kit at samples/ai_toolchain/horizon_model_convert_sample/03_classification/01_mobilenetv1.

## 3. Deployment and Testing

After downloading the .hbm file, you can run the test_MobileNetV1.ipynb Jupyter notebook to test the MobileNetV1 model on the board. To change the test image, download the dataset and place it in the data folder, then update the image path in the Jupyter notebook.

![](./data/image.png)
