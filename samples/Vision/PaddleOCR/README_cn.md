[English](./README.md) | 简体中文

# PaddleOCR 算法简介

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 是百度飞桨基于深度学习的光学字符识别（OCR）工具，依托 PaddlePaddle 框架实现图像文字识别。该工具通过图像预处理、文字检测和文字识别等步骤，实现图像到可编辑文本的转换。PaddleOCR 支持多语言和多字体，适用于复杂场景中的文字提取任务，并支持自定义训练以优化模型效果。

算法论文：[https://arxiv.org/abs/2206.03001](https://arxiv.org/abs/2206.03001)

## 工作流程

1. **图像预处理**：如去噪、尺寸调整等。
2. **文字检测**：使用深度模型检测文字区域，生成检测框。
3. **文字识别**：识别检测框中的文字内容，生成最终输出。

---

## 快速体验

### 环境依赖

```bash
# 安装 PaddlePaddle（S100 上）
pip install paddlepaddle
```

---

## 模型下载与验证

下载 `.hmb` 文件后，可通过 Python/Jupyter 文件进行模型推理。更换测试图片时，需修改脚本中的路径。

---

## 模型转换与量化流程

### 环境准备

```bash
# 克隆仓库并安装依赖
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && python3 -m pip install -e .

# 安装 Paddle2ONNX 与 ONNXRuntime
python3 -m pip install paddle2onnx
python3 -m pip install onnxruntime
```

---

### ONNX 模型导出（以 PP-OCRv3 为例）

```bash
# 下载检测模型
wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && cd ..

# 下载识别模型
wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
cd ./inference && tar xf ch_PP-OCRv3_rec_infer.tar && cd ..
```

```bash
# 转换检测模型
paddle2onnx --model_dir ./inference/ch_PP-OCRv3_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/det_onnx/model.onnx \
--opset_version 19 \
--enable_onnx_checker True

# 转换识别模型
paddle2onnx --model_dir ./inference/ch_PP-OCRv3_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/rec_onnx/model.onnx \
--opset_version 19 \
--enable_onnx_checker True
```

---

## 模型量化

### 数据集准备

使用 [ICDAR2019-LSVT 数据集](https://ai.baidu.com/broad/introduction?dataset=lsvt)

- **数据量**：共 45 万中文街景图像
- **标注类型**：
  - 全标注（文本框+内容）：5 万（2 万测试 + 3 万训练）
  - 弱标注（仅内容）：40 万

下载地址：[点击下载](https://ai.baidu.com/broad/download?dataset=lsvt)

### 校准数据处理

```bash
python process_data.py --src_dir your_dataset --dst_dir det_calibration_data \
--read_mode opencv --cal_img_num 100 --saved_data_type float32
```

---

## 模型编译：检测模型

**配置文件：`yaml_det_configs100.yaml`**

```yaml
model_parameters:
  onnx_model: './../PaddleOCR/inference/det_onnx/modelv3.onnx'
  march: "nash-e"
  layer_out_dump: False
  working_dir: 'model_output'
  output_model_file_prefix: 'cn_PP-OCRv3_det_infer-deploy_640x640_nv12'
  remove_node_type: "Dequantize"

input_parameters:
  input_name: "x"
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  input_shape: '1x3x640x640'
  norm_type: 'data_mean_and_scale'
  mean_value: 123.675 116.28 103.53
  scale_value: 0.01712475 0.017507 0.01742919

calibration_parameters:
  cal_data_dir: './../calibration_data'
  cal_data_type: 'float32'
  calibration_type: 'default'

compiler_parameters:
  compile_mode: 'latency'
  debug: False
  optimize_level: 'O2'
```

```bash
hb_compile -c yaml_det_configs100.yaml
```

---

## 模型编译：识别模型

执行以下步骤生成数据：

```bash
# S100 上运行
python get_rec_calbration_data.py
```

拷贝数据到服务器 docker 中，配置如下：

**配置文件：`yaml_rec_configs100.yaml`**

```yaml
model_parameters:
  onnx_model: './../PaddleOCR/inference/rec_onnx/model_recv3.onnx'
  march: "nash-e"
  layer_out_dump: False
  working_dir: 'model_output'
  output_model_file_prefix: 'cn_PP-OCRv3_rec_infer-deploy_48x320_rgb'
  node_info:
    "p2o.Softmax.0": { 'ON': 'BPU', 'InputType': 'int16', 'OutputType': 'int16' }
    "p2o.Softmax.1": { 'ON': 'BPU', 'InputType': 'int16', 'OutputType': 'int16' }
    "p2o.Softmax.2": { 'ON': 'BPU', 'InputType': 'int16', 'OutputType': 'int16' }

input_parameters:
  input_type_rt: 'featuremap'
  input_layout_rt: 'NCHW'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  input_shape: '1x3x48x320'
  norm_type: 'no_preprocess'

calibration_parameters:
  cal_data_dir: './../calibration_data_rec'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: "set_all_nodes_int16"

compiler_parameters:
  compile_mode: 'latency'
  debug: False
  optimize_level: 'O2'
```

```bash
hb_compile -c yaml_rec_configs100.yaml
```

---

## 性能验证

### 检测模型 (`det`)

```bash
hrt_model_exec perf --model_file cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm
```

- **帧数**：200
- **平均延迟**：1.219ms
- **最大延迟**：16.704ms
- **FPS**：798.575

---

### 识别模型 (`rec`)

```bash
hrt_model_exec perf --model_file cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm
```

- **帧数**：200
- **平均延迟**：2.588ms
- **最大延迟**：17.812ms
- **FPS**：380.525

---

## 精度验证

- **检测模型量化后余弦相似度**  
  ![det_cosine](图片路径)

- **识别模型量化后余弦相似度**  
  ![rec_cosine](图片路径)

---

是否需要我为这份 Markdown 文件生成 `.md` 文件下载？