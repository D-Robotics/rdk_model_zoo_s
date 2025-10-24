[English](./README.md) | 简体中文
# High Accuracy SigLIP on BPU Nash

## Introduction

SigLIP 是一个多模态图像-文本模型, 类似于 CLIP. 它使用独立的图像和文本编码器来生成两种模态的表示. 在VLM模型例如PaliGemma, MiniCPM-V中, VLA模型例如RDT, PI0, OpenVLA中, 均使用SigLIP家族作为视觉编码器, 将图像编码为高维嵌入向量, 供下游结构理解图像模态的信息.

而SigLIP这种ViT类型的视觉编码器在端侧部署有较大挑战, 其LayerNorm结构非常容易溢出导致量化精度不足, 本文针对SigLIP的视觉编码部分(VisionEncoder), 基于Google在HuggingFace上提供的权重, 将其量化为BPU Nash模型. 其中, 利用HBDK4工具集, 保证端侧性能的前提下 对其精度进行深入优化, 其ImageNet-1k验证集Zero-Shot零样本分类精度保持均为100%, COCO2017验证集所有图片的last hidden state隐藏层平均余弦相似度在0.98+, 远超传统PTQ链路的精度, 为RDK S100提供了高精度的端侧视觉编码器组件。

## Usage

对于NVIDIA设备上, 使用带CUDA的PyTorch, 调用HuggingFace的trasformers库, 使用SigLIP模型对图像进行视觉编码例子如代码块1, 可以使用本仓库提供的hbm模型, 在RDK S100 / RDK S100P设备上, 使用BPU加速计算, 代码块2的内容可直接替换代码块1, 其数值一致性请参考文后精度BenchMark.

```python
from transformers import SiglipVisionModel, SiglipProcessor
import cv2
import torch

m = SiglipVisionModel.from_pretrained("siglip-so400m-patch14-384").to(torch.device("cuda:0"))

'''
input_tensor: torch.tensor, float32, NCHW-RGB,  (1, 3, siz, siz), -1.0 ~ +1.0
'''

# Get Image Zero-Shot Embedding Vector
img_vec = model.forward(input_tensor).pooler_output

# Get Image Embedding Tensor (vision_tower)
last_hidden_state = img_vec = model.forward(input_tensor).last_hidden_state
```


```python
from hbm_runtime import HB_HBMRuntime
import cv2
import numpy as np

model = HB_HBMRuntime("bpu-siglip-so400m-patch14-384.hbm")

'''
input_tensor: np.array, float32, NCHW-RGB,  (1, 3, siz, siz), -1.0 ~ +1.0
'''

# Get Image Zero-Shot Embedding Vector
img_vec = model.run({"pooler_output":{'_input_0': input_tensor}})['pooler_output']['_output_0']

# Get Image Embedding Tensor (vision_tower)
last_hidden_state = model.run({"last_hidden_state":{'_input_0': input_tensor}})['last_hidden_state']['_output_0']

```

参考数据预处理函数

```python
def preprocess(image, target_size=384):
    # HWC, BGR, 0~255 -> (1, 3, 384, 384), RGB, -1~1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    image_padded = cv2.copyMakeBorder(
        image_resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=[127, 127, 127]  # 灰色边框
    )
    # 4. HWC -> CHW -> NCHW
    image_chw = np.transpose(image_padded, (2, 0, 1))  # HWC -> CHW
    image_nchw = np.expand_dims(image_chw, axis=0)     # CHW -> NCHW (batch=1)
    # 5. 归一化: / 127.5 - 1 → [-1, 1]
    image_normalized = image_nchw.astype(np.float32)
    image_normalized = image_normalized / 127.5 - 1.0
    return image_normalized
    # return torch.from_numpy(image_normalized)  # Optional

import cv2
import numpy as np


img = cv2.imread("test_img.jpg")
input_tensor = preprocess(img)
```

## Downloads
| Model Name (Packed)            | Support BPU    |
|--------------------------------|----------------|
| [bpu-siglip-base-patch16-224](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-224.hbm)        | Nash-e, Nash-m |
| [bpu-siglip-base-patch16-384](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-384.hbm)         | Nash-e, Nash-m |
| [bpu-siglip-base-patch16-512](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-512.hbm)         | Nash-e, Nash-m |
| [bpu-siglip-large-patch16-256](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-256.hbm)        | Nash-e, Nash-m |
| [bpu-siglip-large-patch16-384](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-384.hbm)        | Nash-e, Nash-m |
| [bpu-siglip-so400m-patch14-224](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-224.hbm)       | Nash-e, Nash-m |
| [bpu-siglip-so400m-patch14-384](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-384.hbm)       | Nash-e, Nash-m |
| [bpu-siglip-so400m-patch16-256-i18n](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch16-256-i18n.hbm)  | Nash-e, Nash-m |


## Reference BenchMark

### Performance


| Model Name (last hidden state) | Input Size    | Embedding Size | Params <br/> total / vision | Inference Time <br/> RDK S100 |Inference Time <br/> RDK S100P |
|--------------------------------|---------------|----------------|----------------|----------|-----------|
| siglip-base-patch16-224        | (1,3,224,224) | (1,196,768)    | 0.2 B / 0.09 B | 26.0 ms  | 18.3 ms  |
| siglip-base-patch16-384        | (1,3,384,384) | (1,576,768)    | 0.2 B / 0.09 B | 45.9 ms  | 31.7 ms  |
| siglip-base-patch16-512        | (1,3,512,512) | (1,1024,768)   | 0.2 B / 0.09 B | 80.8 ms  | 55.3 ms  |
| siglip-large-patch16-256       | (1,3,256,256) | (1,256,1024)   | 0.7 B / 0.32 B | 67.6 ms  | 46.5 ms  |
| siglip-large-patch16-384       | (1,3,384,384) | (1,576,1024)   | 0.7 B / 0.32 B | 131.3 ms | 90.5 ms  |
| siglip-so400m-patch14-224      | (1,3,224,224) | (1,256,1152)   | 0.9 B / 0.43 B | 88.6 ms  | 61.4 ms  |
| siglip-so400m-patch14-384      | (1,3,384,384) | (1,729,1152)   | 0.9 B / 0.43 B | 254.2 ms | 174.5 ms |
| siglip-so400m-patch16-256-i18n | (1,3,256,256) | (1,256,1152)   | 1.0 B / 0.43 B | 88.3 ms  | 61.1 ms  |



| Model Name (pooler output)     | Input Size    | Embedding Size | Params <br/> total / vision | Inference Time <br/> RDK S100 |Inference Time <br/> RDK S100P |
|--------------------------------|---------------|----------------|----------------|----------|----------|
| siglip-base-patch16-224        | (1,3,224,224) | (1,1,768)      | 0.2 B / 0.09 B | 26.8 ms  | 18.8 ms  |
| siglip-base-patch16-384        | (1,3,384,384) | (1,1,768)      | 0.2 B / 0.09 B | 46.7 ms  | 32.3 ms  |
| siglip-base-patch16-512        | (1,3,512,512) | (1,1,768)      | 0.2 B / 0.09 B | 81.7 ms  | 55.8 ms  |
| siglip-large-patch16-256       | (1,3,256,256) | (1,1,1024)     | 0.7 B / 0.32 B | 68.8 ms  | 47.2 ms  |
| siglip-large-patch16-384       | (1,3,384,384) | (1,1,1024)     | 0.7 B / 0.32 B | 132.5 ms | 91.4 ms  |
| siglip-so400m-patch14-224      | (1,3,224,224) | (1,1,1152)     | 0.9 B / 0.43 B | 89.8 ms  | 62.2 ms  |
| siglip-so400m-patch14-384      | (1,3,384,384) | (1,1,1152)     | 0.9 B / 0.43 B | 255.7 ms | 175.5 ms |
| siglip-so400m-patch16-256-i18n | (1,3,256,256) | (1,1,1152)     | 1.0 B / 0.43 B | 89.6 ms  | 61.9 ms  |

### Performance Test Instructions

1. BPU延迟使用以下命令在板端进行测试, 每个hbm模型由last_hidden_state和pooler_output两个子模型pack而成, 两个子模型共享权重.
```bash
hrt_model_exec perf --thread_num 1 --model_name last_hidden_state --model_file <*.hbm>
```
2. 测试板卡为最佳状态.

- S100P的状态为最佳状态: CPU为6 × A78AE @ 2.0GHz, 全核心Performance调度, BPU为1 × Nash-m @ 1.5GHz, 128TOPS @ int8.
- S100的状态为最佳状态: CPU为6 × A78AE @ 1.5GHz, 全核心Performance调度, BPU为1 × Nash-e @ 1.0GHz, 80TOPS @ int8.
```
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"
```


### Accuracy


| Model Name (last hidden state) | PyTorch TOP1 / TOP5 | BPU TOP1 / TOP5 |
|--------------------------------|---------------------|-----------------|
| siglip-base-patch16-224        | 0.7123 / 0.9143     | 0.7118 / 0.9144 |
| siglip-base-patch16-384        | 0.7411 / 0.9318     | 0.7418 / 0.9319 |
| siglip-base-patch16-512        | 0.7490 / 0.9343     | 0.7482 / 0.9340 |
| siglip-large-patch16-256       | 0.7490 / 0.9238     | 0.7490 / 0.9242 |
| siglip-large-patch16-384       | 0.7584 / 0.9252     | 0.7595 / 0.9256 |
| siglip-so400m-patch14-224      | 0.7659 / 0.9361     | 0.7651 / 0.9357 |
| siglip-so400m-patch14-384      | 0.7872 / 0.9433     | 0.7893 / 0.9447 |
| siglip-so400m-patch16-256-i18n | 0.7678 / 0.9395     | 0.7668 / 0.9397 |



| Model Name (pooler output)     | Cosine Similarity <br/> mean (min ~ max), %1low | MSE <br/> mean (min ~ max), %1low | 
|--------------------------------|--------------------------------|--------------------------------|
| siglip-base-patch16-224        | 0.991 ( 0.951 ~ 0.997 ), 0.980 | 0.087 ( 0.024 ~ 0.471 ), 0.039 | 
| siglip-base-patch16-384        | 0.989 ( 0.960 ~ 0.997 ), 0.977 | 0.113 ( 0.029 ~ 0.409 ), 0.050 | 
| siglip-base-patch16-512        | 0.987 ( 0.956 ~ 0.995 ), 0.974 | 0.142 ( 0.045 ~ 0.507 ), 0.067 | 
| siglip-large-patch16-256       | 0.990 ( 0.933 ~ 0.997 ), 0.974 | 0.069 ( 0.018 ~ 0.497 ), 0.024 | 
| siglip-large-patch16-384       | 0.985 ( 0.900 ~ 0.995 ), 0.965 | 0.111 ( 0.034 ~ 0.775 ), 0.048 | 
| siglip-so400m-patch14-224      | 0.984 ( 0.850 ~ 0.995 ), 0.961 | 0.104 ( 0.028 ~ 1.038 ), 0.041 | 
| siglip-so400m-patch14-384      | 0.980 ( 0.859 ~ 0.993 ), 0.957 | 0.140 ( 0.040 ~ 1.093 ), 0.059 | 
| siglip-so400m-patch16-256-i18n | 0.984 ( 0.878 ~ 0.996 ), 0.959 | 0.082 ( 0.018 ~ 0.570 ), 0.030 | 

### Accuracy Test Instructions

1. last_hidden_state语义一致性验证中, 数据集为COCO2014数据集的验证集, val验证集的图片数量为5,000张, 取的精度指标为余弦相似度(Cosine Similarity)和均方误差(MSE), 主要用于验证定点模型和浮点模型输出的高位嵌入张量的语义一致性.
2. pooler_output零样本分类精度验证中,的数据集为Ima geNet-1k数据集的验证集合, val验证集的图片数量为50,000张, 取的精度指标为TOP1和TOP5正确率, 主要用于验证定点模型和浮点模型在分类这个下游任务的行为一致性.
3. last_hidden_state语义一致性验证和pooler_output零样本分类精度验证, 图像前处理均为(127,127,127)颜色letter box, 浮点模型和BPU模型的前处理一致.


## Model Convert

由于这里是旁门左道的方法解决相似问题, 转化方法难以整理和开源, 而SigLIP基本是冻结的, 所以这里尽可能多的转化了Google开源的权重, 供大家使用.

## Contributers
```
Cauchy @吴超
```