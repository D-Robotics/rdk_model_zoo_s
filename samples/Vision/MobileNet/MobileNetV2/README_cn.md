[English](./README.md) | 简体中文

# MobileNetV2

- [MobileNetV2](#mobilenetv2)
  - [1. 简介](#1-简介)
  - [2. 模型下载](#2-模型下载)
    - [选项一](#选项一)
    - [选项二](#选项二)
  - [3. 部署测试](#3-部署测试)
  - [4. 量化实验](#4-量化实验)
    - [数据集准备](#数据集准备)
    - [校准数据处理](#校准数据处理)
    - [模型检查](#模型检查)
    - [模型编译](#模型编译)
    - [模型推理](#模型推理)

## 1. 简介

- **论文地址**: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

- **Github 仓库**: [timm/docs/models/mobilenet-v2.md at master · pprp/timm (github.com)](https://github.com/pprp/timm/blob/master/docs/models/mobilenet-v2.md)


Mobilenetv2 是对 [Mobilenet](../MobileNetV1/README_cn.md) 的改进，同样是一种轻量级的神经网络。Mobilenetv2 为了防止非线性层ReLU损失一部分信息，引入了**线性瓶颈层(Linear Bottleneck)**；另外借鉴 Resnet 等一系列网络采用了残差网络得到了很好的效果，作者结合点态卷积的特点，提出了**倒残差 (Inverted Residual)结构**。论文在ImageNet classification, MS COCO object detection, VOC image segmentation上做了对比实验，验证了该架构的有效性。

Mobilenetv2 在深度卷积前新加了一个点态卷积。这么做的原因，是因为深度卷积由于本身的计算特性，决定其自身没有改变通道数的能力，上一层给它多少通道，它就只能输出多少通道。所以如果上一层给的通道数本身很少的话，深度卷积也只能在低维空间提特征，因此效果不够好。为了改善这个问题，Mobilenetv2 给每个深度卷积之前都配备了一个点态卷积专门用来升维。

![](./data/seperated_conv.png)
![](./data/mobilenetv2_architecture.png)


## 2. 模型下载

### 选项一

可以使用脚本 [download_1.sh](./model/download_1.sh) 一键下载此模型结构的 .hbm 模型文件，方便直接更换模型。或者使用以下命令行进行下载：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv2_224x224_nv12_1.hbm
```

此模型是由地平线参考算法进行模型量化后得到的产出物。

模型转换使用的是Caffe model: https://github.com/shicai/MobileNet-Caffe

若需要 MobileNetV1 模型量化转换步骤，可以参考 MobileNet 其他模型的转换步骤或是直接使用 OE 开发包中的 samples/ai_toolchain/horizon_model_convert_sample/03_classification/01_mobilenetv2

### 选项二

**.hbm 文件下载**：

可以使用脚本 [download.sh](./model/download.sh) 一键下载此模型结构的 .hbm 模型文件，方便直接更换模型。或者使用以下命令行进行下载：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv2_224x224_nv12.hbm
```

**ONNX文件下载**：

onnx 模型使用的是 timm 库 (PyTorch Image Models) 中的模型进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

安装完必要的库后可以使用python文件夹下的 [get_mobilenetv2_onnx.py](python/get_mobilenetv2_onnx.py) 脚本完成onnx文件的下载

* 注：需要配置终端代理并使用以下命令登录 huggingface
```shell
huggingface-cli login
```

* 若不想配置终端代理可选择手动前往 [timm/mobilenetv2_100.ra_in1k](https://huggingface.co/timm/mobilenetv2_100.ra_in1k) 下载模型 并使用 [python/timm2onnx.py](python/timm2onnx_local.py) 脚本完成onnx转换
  
完成onnx文件导出后脚本会输出模型 input, mean, std, path, parameters 等信息，格式如下：

```shell
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv2_100.onnx
Total number of parameters in the model: 3487818
```

## 3. 部署测试

在下载完毕 .hbm 文件后，可以执行 'test_mobilenetv2.ipynb' 或 pyhton文件夹中的 's100_inference.py' ，在板端实际运行体验实际测试效果。

* 注：若使用地平线参考算法，需要修改 `classification_postprocess_info.use_softmax = False`

若需要更改测试图片，可额外下载数据集后，放入到data文件夹下并更改 jupyter 文件或 python脚本 中图片的路径

![inference](data/image.png)

* 部署性能测试：
```shell
hrt_model_exec perf --model_file ./model/mobilenetv2_224x224_nv12.hbm \
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
- Frame totally latency is: 80.924 ms
- Average latency is: 0.405 ms
- Frame rate is: 2345.381 FPS
```
thread_num = 3
```
Running condition:
- Thread number is: 3
- Frame count is: 200
Perf result:
- Frame totally latency is: 113.025 ms
- Average latency is: 0.565 ms
- Frame rate is: 5026.515 FPS
```


## 4. 量化实验

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
hb_compile --model mobilenetv2_100.onnx --march nash-e
```

### 模型编译

模型验证通过后可以通过校准数据集进行模型量化编译，参考 yaml 文件已提供在 yaml 文件夹中，运行以下命令：

```shell
hb_compile --config yaml/mobilenetv2_config.yaml
```

完成模型编译后可在 model_output 目录下发现编译产物，需要上板使用的为 mobilenetv2_224x224_nv12.hbm 

* 模型量化后余弦相似度

```shell
 +------------+-------------------+------------------+
 | TensorName | Calibrated Cosine | Quantized Cosine |
 +------------+-------------------+------------------+
 | output     | 0.993383          | 0.988877         |
 +------------+-------------------+------------------+
```

* 工具链给出的性能参考

```bash
Summary:
FPS (1 core): 4968.89
latency: 0.2 ms (201.3 us)
BPU conv original OPs per run: 601,548,544
```

### 模型推理

在 python 目录下提供了在 X86 平台和 S100 平台快速进行推理的 demo， 其中：
* [x86_inference.py](python/x86_inference.py) 支持 ONNX , HBIR(.bc) 和 HBM 格式在 X86 平台的推理。
* [s100_inference.py](python/s100_inference.py) 支持 HBM 格式在板端的推理。

x86_inference.py 需要通过 -m , -i 传入模型路径和图像路径，示例
```shell
python3 python/x86_inference.py -m model_output/mobilenetv2_224x224_nv12_quantized_model.bc -i data/zebra_cls.jpg
```

s100_inference.py 需要修改 main 函数中模型和图像路径