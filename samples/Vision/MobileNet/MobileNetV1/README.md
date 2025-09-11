English | [简体中文](./README_cn.md)

# MobileNetV1

- [MobileNetV1](#mobilenetv1)
  - [1. Introduction](#1-introduction)
  - [2. Model Performance Data](#2-model-performance-data)
  - [3. Model Download](#3-model-download)
  - [4. Deployment and Testing](#4-deployment-and-testing)

## 1. Introduction

- **Paper**: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

- **GitHub Repository**: [models/research/slim/nets/mobilenet_v1.md at master · tensorflow/models (github.com)](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

MobileNetV1 introduces a lightweight neural network designed for embedded devices. It utilizes depthwise separable convolutions to construct efficient deep neural networks. The core idea is to decompose standard convolutions into **depthwise convolutions** and **pointwise convolutions**. By separating standard convolutions, the number of intermediate feature maps is reduced, effectively decreasing the number of network parameters.

![](./data/depthwise&pointwise.png)

**Key Features of MobileNetV1:**

- **Depthwise Separable Convolutions**: MobileNet is based on depthwise separable convolutions, a form of **factorized convolution** that splits a standard convolution into a depthwise convolution and a 1×1 convolution called pointwise convolution, first introduced in InceptionV3.
- **Hyperparameters**: The width multiplier $\alpha$ and resolution multiplier $\rho$ are used to reduce computational cost and model size.

## 2. Model Performance Data

The following table shows the performance data tested on the RDK S100.

| Model         | Input Size (pixels) | Classes | Params (M) | FP Top-1 | Quantized Top-1 | Latency/Throughput (Single Thread) | Latency/Throughput (Multi Thread) | FPS   |
| ------------- | ------------------ | ------- | ---------- | -------- | --------------- | ---------------------------------- | ---------------------------------- | ----- |
| MobileNetV1   | 224x224            | 1000    | 4.7        | 70.8     | -               | -                                  | -                                  | -     |

Notes:
1. The S100 was tested in its optimal state.
2. Single-thread latency refers to the latency for a single frame, single thread, and single BPU core—the ideal scenario for BPU inference of a single task.
3. FP/Quantized Top-1: FP Top-1 refers to the Top-1 inference accuracy of the ONNX model before quantization, while Quantized Top-1 refers to the actual inference accuracy after quantization.

## 3. Model Download

**.hbm File Download:**

Go to the model folder and use the following command to download the MobileNetV1 model:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/mobilenetv1_224x224_nv12.hbm
```

This model is the quantized output produced by Horizon's reference algorithm.

Model conversion uses the Caffe model: https://github.com/shicai/MobileNet-Caffe

For quantization and conversion steps of MobileNetV1, refer to the conversion steps of other MobileNet models or directly use the samples in the OE development kit at samples/ai_toolchain/horizon_model_convert_sample/03_classification/01_mobilenetv1.

## 4. Deployment and Testing

After downloading the .hbm file, you can run the test_MobileNetV1.ipynb Jupyter notebook to test the MobileNetV1 model on the board. 

To change the test image, download the dataset and place it in the data folder, then update the image path in the Jupyter notebook


![](./data/image.png)


A demo for fast inference on both X86 and S100 platforms is provided in the `python` directory:

- [`x86_inference.py`](python/x86_inference.py): Supports inference on X86 platforms using ONNX, HBIR (.bc), and HBM formats, as well as accuracy validation on the validation dataset.
- [`s100_inference.py`](python/s100_inference.py): Supports inference on the board using the HBM format.

For `x86_inference.py`, specify the model and image paths using the `-m` and `-i` options. Example:

```shell
python3 python/x86_inference.py -m model_output/mobilenetv1_224x224_nv12_quantized_model.bc -i data/zebra_cls.jpg
```

To enable accuracy validation with `x86_inference.py`, use the `--validate` option. Example:

```shell
python3 python/x86_inference.py -m model_output/mobilenetv1_224x224_nv12_quantized_model.bc --validate -d ../../../imagenet/val -l ../../../imagenet/val.txt
```

