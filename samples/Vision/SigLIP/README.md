English| [简体中文](./README_cn.md)


# High Accuracy SigLIP on BPU Nash

## Introduction

SigLIP is a multimodal image-text model similar to CLIP. It employs separate image and text encoders to generate representations for both modalities. SigLIP-family models are widely used as vision encoders in VLMs such as PaliGemma and MiniCPM-V, and in VLA models such as RDT, PI0, and OpenVLA, where they encode images into high-dimensional embedding vectors for downstream components to understand visual information.

However, deploying ViT-type vision encoders like SigLIP on edge devices poses significant challenges. The LayerNorm structure in SigLIP is particularly prone to numerical overflow, leading to insufficient quantization accuracy. This paper presents a quantization of the SigLIP vision encoder (VisionEncoder) into a BPU Nash model using the official weights provided by Google on Hugging Face. Leveraging the HBDK4 toolkit, we achieve deep accuracy optimization while ensuring on-device performance. Our quantized model maintains 100% zero-shot classification accuracy on the ImageNet-1k validation set and achieves an average cosine similarity of over 0.98 for the last hidden state across all images in the COCO2017 validation set—significantly outperforming traditional PTQ pipelines. This delivers a high-accuracy vision encoder component for the RDK S100 platform.

## Usage

On NVIDIA devices, using PyTorch with CUDA, you can invoke the Hugging Face Transformers library to perform visual encoding with the SigLIP model, as shown in Code Block 1. Alternatively, you can use the .hbm model provided in this repository to accelerate computation via the BPU on RDK S100 / RDK S100P devices. Code Block 2 can directly replace Code Block 1, with numerical consistency details provided in the accuracy benchmark section below.


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

Reference Pre Process Function

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
        value=[127, 127, 127]
    )
    # 4. HWC -> CHW -> NCHW
    image_chw = np.transpose(image_padded, (2, 0, 1))  # HWC -> CHW
    image_nchw = np.expand_dims(image_chw, axis=0)     # CHW -> NCHW (batch=1)
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

1. BPU latency is tested on the board using the following command. Each HBM model is packed from two sub-models, last_hidden_state and pooler_output, which share weights.
bash
hrt_model_exec perf --thread_num 1 --model_name last_hidden_state --model_file <*.hbm>

2. The test boards are configured in their optimal performance states.
S100P is set to its optimal state: CPU with 6 × A78AE @ 2.0 GHz running under full-core Performance scheduling, and BPU with 1 × Nash-m @ 1.5 GHz, delivering 128 TOPS at int8.
S100 is set to its optimal state: CPU with 6 × A78AE @ 1.5 GHz running under full-core Performance scheduling, and BPU with 1 × Nash-e @ 1.0 GHz, delivering 80 TOPS at int8.

The following commands ensure the system operates in maximum performance mode:
bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"



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

1. In the last_hidden_state semantic consistency verification, the dataset used is the validation set of the COCO2014 dataset, which contains 5,000 images. The evaluation metrics employed are Cosine Similarity and Mean Squared Error (MSE), primarily used to verify the semantic consistency of high-order embedding tensors output by the fixed-point model and the floating-point model.
2. In the pooler_output zero-shot classification accuracy verification, the dataset used is the validation set of ImageNet-1k, containing 50,000 images. The evaluation metrics are Top-1 and Top-5 accuracy, mainly used to verify the behavioral consistency between the fixed-point model and the floating-point model on the downstream task of image classification.
3. For both the last_hidden_state semantic consistency verification and the pooler_output zero-shot classification accuracy verification, image preprocessing involves a (127, 127, 127) color letter box. The preprocessing procedures for both the floating-point model and the BPU model are identical.





## Model Convert

Since this is a roundabout way to solve similar problems, the conversion methods are difficult to organize and open source. SigLIP is basically frozen, so we have converted as much of Google's open source weight as possible here for everyone to use.

## Contributers
```
Cauchy @吴超
```