English| [简体中文](./README_cn.md)

# Introduction

In the field of robotics, bridging the gap between simulation and real-world physics remains a key challenge for achieving precise motion control. Building upon the optimized framework of *Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills*, DiGua Robotics, in collaboration with Horizon Robotics Lab, has successfully deployed the enhanced ASAP framework on the RDK S100 platform.

The ASAP framework adopts a two-stage training strategy combined with a Delta action model, enabling efficient whole-body control of humanoid robots in real-world environments. This significantly mitigates the dynamics mismatch between simulation and reality, resulting in excellent and stable performance on physical machines.

The major breakthrough lies in drastically reducing resource consumption during deployment: BPU-based inference only occupies **2%** of processing resources. Compared with CPU-only inference, CPU utilization is reduced by **250%**, allowing more computing resources to be allocated to advanced tasks such as vision detection, object recognition, path planning, and intelligent decision-making. This substantially enhances the system’s overall capabilities.

The optimized ASAP framework is not yet open-sourced. It is expected to be released in **June**, along with the full training and deployment process. Stay tuned! This document only covers the **quantization process** for now.

---

# Model Quantization

## Environment Setup

- Pull the S100 toolchain Docker image. *(Please contact engineers to obtain the Docker image and OE package.)*
- Extract the quantization folder (attached at the end of this document). It contains the required configuration files and calibration data.
- Place the exported ONNX model inside the extracted folder. Ensure that the model name matches the name in the config file.
- Mount the folder to the Docker container and start the container:

```bash
sudo docker run -it --entrypoint="/bin/bash" \
-v /your/host/folder/path:/g1_model_convert_s100 \
<docker-image-id>
````

## Quantization and Compilation

```bash
# Navigate to the mounted folder
cd /g1_model_convert_s100

# Quantize the model. This will create a folder called 'model_output' containing the .bin model.
hb_compile -c dance_bpu.yaml
```
![](https://developer.d-robotics.cc/api/v1/static/imgData/1748580922266.jpg)
> ⚠️ Without further optimization, the quantized model may not reach high accuracy. It is recommended to decompose the intermediate float model (`*_optimized_float_model`) generated during quantization.

### Operator Decomposition for Higher Precision

Use the provided script to split the optimized float model. Modify the model name and replace the convolution operator name with the one highlighted in the reference image.

```bash
python split_float_model.py
```

Then quantize the decomposed float model again using the original config file:

```bash
hb_compile -c dance_bpu.yaml
```
![](https://developer.d-robotics.cc/api/v1/static/imgData/1748580943990.jpg)
---

# Performance Demo

Watch the performance video on Bilibili:
[从仿真到实机无缝迁移！地瓜机器人基于RDK S100复现CoRL获奖论文仿生步态](https://www.bilibili.com/video/BV154d2YTEn1/)

---

# Attachments

The quantization folder is available via Baidu Cloud:

* **File**: `g1_model_convert_s100.zip`
* **Link**: [https://pan.baidu.com/s/1DMyqMGk1IcE9tuk9OkM30w?pwd=fkt3](https://pan.baidu.com/s/1DMyqMGk1IcE9tuk9OkM30w?pwd=fkt3)
* **Extraction Code**: `fkt3`

---

# Acknowledgements

Special thanks to **Horizon Robotics Lab** for their strong support!
Thanks also to **Yucheng Wang**, **Kaihui Wang**, and **Maiyue Chen** for their tremendous contributions!
