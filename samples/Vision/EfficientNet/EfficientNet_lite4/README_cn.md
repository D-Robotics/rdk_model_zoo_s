[English](./README.md) | 简体中文

# EfficientNet_lite4

- [EfficientNet\_lite4](#efficientnet_lite4)
  - [1. 简介](#1-简介)
  - [2. 模型性能数据](#2-模型性能数据)
  - [3. 模型下载](#3-模型下载)
  - [4. 部署测试](#4-部署测试)
  - [5. 量化实验](#5-量化实验)
    - [数据集准备](#数据集准备)
    - [校准数据处理](#校准数据处理)
    - [模型检查](#模型检查)
    - [模型编译](#模型编译)
    - [模型推理](#模型推理)

## 1. 简介

- **论文地址**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

- **Github 仓库**: [tensorflow/tpu/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

![](./data/EfficientNet_architecture.png)

**EfficientNet 模型特点**：

- **复合缩放方法**：通过统一调整分辨率、深度和宽度三个维度，最大化模型性能与效率，而不是单独调整某一个维度。
- **AutoML 技术**：利用 AutoML 自动搜索最佳网络参数组合，使模型在保持高准确率的同时，显著减少参数量和计算资源消耗。
- **高效且轻量化的网络结构**：通过优化的复合缩放策略，EfficientNet 实现了较少的参数量和更快的推理速度，但在处理较大模型时可能会消耗更多显存。

EfficientNet 是一种平衡卷积神经网络（CNN）分辨率、深度和宽度的创新网络结构。其核心特点在于通过复合缩放（Compound Scaling）方法，系统性地优化网络性能与效率。传统的网络设计通常只对网络的单一维度进行调整，例如宽度、深度或输入图像的分辨率。而 EfficientNet 则通过一个简单且高效的复合系数，统一调整这三个关键维度。这种方法通过网格搜索来确定每个维度的最佳比例，从而在给定的资源限制下，最大化网络的整体性能。

EfficientNet 结合了 AutoML 技术，自动搜索出最佳的网络参数组合，使模型在保持高准确率的同时，大幅减少参数量和计算资源。例如，EfficientNet-B7 在 ImageNet 数据集上实现了 84.3% 的 top-1 准确率，且其参数量仅为 GPipe 的 1/8.4，推理速度提升了 6.1 倍。

EfficientNet-lite 是一组适用于移动设备 / 物联网的图像分类模型。值得注意的是，虽然 EfficientNet-EdgeTPU 是专门为 Coral EdgeTPU 设计的，但这些 EfficientNet-lite 模型在所有移动 CPU/GPU/EdgeTPU 上都能良好运行。

由于边缘设备的需求，我们在原始 EfficientNets 的基础上主要做了以下改动。

* 移除挤压激励（SE）模块：某些移动加速器对 SE 模块支持不佳。
* 将所有 Swish 激活函数替换为 RELU6：以便于后量化。
* 在放大模型时固定网络的起始和结束部分：以保持模型小巧且快速。
  
以下是各模型的检查点，以及它们的准确率、参数数量、浮点运算次数，还有 Pixel4 设备在 CPU/GPU/EdgeTPU 上的延迟。
|**Model** | **params** | **MAdds** | **FP32 accuracy** | **FP32 CPU  latency** | **FP32 GPU latency** | **FP16 GPU latency** |**INT8 accuracy** | **INT8 CPU latency**  | **INT8 TPU latency**|
|------|-----|-------|-------|-------|-------|-------|-------|-------|-------|
|efficientnet-lite4 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz) | 4.7M | 407M |  75.1% |  12ms | 9.0ms | 6.0ms  | 74.4% |  6.5ms | 3.8ms |
|efficientnet-lite1 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz) | 5.4M | 631M |  76.7% |  18ms | 12ms | 8.0ms  |  75.9% | 9.1ms | 5.4ms |
|efficientnet-lite2 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite2.tar.gz) | 6.1M | 899M |  77.6% |  26ms | 16ms | 10ms | 77.0% | 12ms | 7.9ms |
|efficientnet-lite3 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite3.tar.gz) | 8.2M | 1.44B |  79.8% |  41ms | 23ms | 14ms  | 79.0% | 18ms | 9.7ms |
|efficientnet-lite4 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz) |13.0M | 2.64B |  81.5% |  76ms | 36ms | 21ms  | 80.2% | 30ms | - |

## 2. 模型性能数据

以下表格是在 RDK S100 上实际测试得到的性能数据


| 模型          | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率     |
| -----------  | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| EfficientNet_lite4   | 380x380 | 1000 | 13.0    | 81.5 | 80.1 | 0.915 ms       | 1.979 ms        | 1487.055 FPS |


说明: 
1. S100的状态为最佳状态
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

## 3. 模型下载

**.hbm 文件下载**：

可以使用脚本 [download.sh](./model/download.sh) 一键下载此模型结构的 .hbm 模型文件，方便直接更换模型。或者使用以下命令行进行下载：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/EfficientNet/efficientnet_lite4_380x380_nv12.hbm
```

**ONNX文件下载**：

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

安装完必要的库后可以使用python文件夹下的 [get_efficientnet_lite4_onnx.py](python/get_efficientnet_lite4_onnx.py) 脚本完成onnx文件的下载

* 注：需要配置终端代理并使用以下命令登录 huggingface
```shell
huggingface-cli login
```

* 若不想配置终端代理可选择手动前往 [timm/tf_efficientnet_lite4.in1k](https://huggingface.co/timm/tf_efficientnet_lite4.in1k) 下载模型 并使用 [python/timm2onnx.py](python/timm2onnx_local.py) 脚本完成onnx转换
  
完成onnx文件导出后脚本会输出模型 input, mean, std, path, parameters 等信息，格式如下：

```shell
input: (3, 380, 380)
mean (0.5, 0.5, 0.5)
std (0.5, 0.5, 0.5)
Simplified model is valid.
Simplified model saved to tf_efficientnet_lite4.onnx
Total number of parameters in the model: 12950386
```

## 4. 部署测试

在下载完毕 .hbm 文件后，可以执行 'test_EfficientNet_lite4.ipynb' 或 'python'文件夹中的 's100_inference.py' ，在板端实际运行体验实际测试效果。

若需要更改测试图片，可额外下载数据集后，放入到data文件夹下并更改 jupyter 文件或 python脚本 中图片的路径

![inference](data/image.png)

* 部署性能测试：
```shell
hrt_model_exec perf --model_file ./model/efficientnet_lite4_380x380_nv12.hbm \
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
- Frame totally latency is: 183.056 ms
- Average latency is: 0.915 ms
- Frame rate is: 1064.339 FPS
```
thread_num = 3
```
Running condition:
- Thread number is: 3
- Frame count is: 200
Perf result:
- Frame totally latency is: 395.802 ms
- Average latency is: 1.979 ms
- Frame rate is: 1487.055 FPS
```


## 5. 量化实验

### 数据集准备

模型使用的数据集为 [ImageNet](https://image-net.org/)
* 数据集：ILSVRC2012

| 数据集名称 | 分类数量 | 图片数量 |
| -- | -- | -- |
| ILSVRC2012训练集 | 1000个分类 | 约120万张图片 |
| ILSVRC2012验证集 | 1000个分类 | 5万张图片 |
| ILSVRC2012测试集 | 1000个分类 | 10万张图片 |

我们建议您将下载的数据集解压成如下结构

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

### 校准数据处理

准备好数量为100的校准数据集后，运行以下命令生成校准数据，校准数据保存在 /calibration_data_rgb 目录下:

```shell
python3 python/get_calibration_data.py
```

### 模型检查

在准备好onnx模型后，可以通过以下命令快速完成模型验证：

```shell
hb_compile --model tf_efficientnet_lite4.onnx --march nash-e
```

### 模型编译

模型验证通过后可以通过校准数据集进行模型量化编译，参考 yaml 文件已提供在 yaml 文件夹中，运行以下命令：

```shell
hb_compile --config yaml/efficientnet_lite4_config.yaml
```

完成模型编译后可在 model_output 目录下发现编译产物，需要上板使用的为 efficientnet_lite4_380x380_nv12.hbm 

* 模型量化后余弦相似度

```shell
 +------------+-------------------+------------------+
 | TensorName | Calibrated Cosine | Quantized Cosine |
 +------------+-------------------+------------------+
 | output     | 0.997189          | 0.997863         |
 +------------+-------------------+------------------+
```

* 工具链给出的性能参考

```bash
Summary:
FPS (1 core): 4258.32
latency: 0.23 ms (234.8 us)
BPU conv original OPs per run: 770,375,104
```

### 模型推理

在 python 目录下提供了在 X86 平台和 S100 平台快速进行推理的 demo， 其中：
* [x86_inference.py](python/x86_inference.py) 支持 ONNX , HBIR(.bc) 和 HBM 格式在 X86 平台的推理。
* [s100_inference.py](python/s100_inference.py) 支持 HBM 格式在板端使用新的 HB_HBMRuntime API 进行推理。

x86_inference.py 需要通过 -m , -i 传入模型路径和图像路径，示例

```shell
python3 python/x86_inference.py -m model_output/efficientnet_lite4_380x380_nv12_quantized_model.bc -i data/zebra_cls.jpg
```

x86_inference.py 使用精度验证需要设置 --validate 启动精度验证模式，示例

```shell
python3 python/x86_inference.py -m model_output/efficientnet_lite4_380x380_nv12_quantized_model.bc --validate -d ../../../imagenet/val -l ../../../imagenet/val.txt
```
