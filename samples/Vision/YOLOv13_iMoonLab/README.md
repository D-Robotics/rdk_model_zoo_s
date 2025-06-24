[English](./README.md) | 简体中文

# Ultralytics YOLO Detect

## Abstract
```bash
D-Robotics OpenExplore Version: >= 3.2.0
```

## Support Models: 

```bash
- YOLOv13 - Detect
```

## YOLOv13介绍

<p align="center">
    <img src="source/imgs/icon.png" width="110" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center">YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception</h2>


  
<div align="center">
    <img src="source/imgs/framework.png" width="96%" height="96%">
</div>

YOLOv13 is a new generation of real-time object detection model developed by the Intelligent Media and Cognition Laboratory at Tsinghua University, featuring excellent performance and efficiency. Its core technologies include HyperACE (Hypergraph-based Adaptive Correlation Enhancement), FullPAD (Full-Pipeline Aggregation-and-Distribution Paradigm), and lightweight convolution replacement. HyperACE explores high-order correlations among pixels through a hypergraph structure, enhancing multi-scale feature fusion. FullPAD achieves fine-grained information flow and representational synergy across the entire network pipeline. The lightweight design reduces computational cost while maintaining the receptive field, thereby accelerating inference speed. Experiments show that YOLOv13 performs exceptionally well on the COCO dataset, outperforming existing models in terms of accuracy, speed, and parameter efficiency.

Reference: 

1. https://github.com/iMoonLab/yolov13/

2. https://www.gaoyue.org/


## Quick Experience

```bash
# Download RDK S100 Model Zoo
$ git clone https://github.com/D-Robotics/rdk_model_zoo_s.git

# Make Sure your are in this file
$ cd samples/Vision/YOLOv13_iMoonLab

# Check your workspace
$ tree -L 2
.
|-- README.md     # English Document
|-- README_cn.md  # Chinese Document
|-- py
|   |-- eval_ultralytics_YOLO_Detect_YUV420SP.py # Advance Evaluation
|   `-- ultralytics_YOLO_Detect_YUV420SP.py      # Quick Start
`-- source
    |-- imgs
    |-- reference_hbm_models    # Reference HBM Models
    |-- reference_logs          # Reference logs
    `-- reference_yamls         # Reference yaml configs
```

Compile the temporary Python interface for the BPU. If there is an error during compilation, please refer to the README of the following repository to update the dynamic libraries and header files of the OpenExplore package.

```bash
https://github.com/WuChao-2024/pyCauchyKesai/blob/main/README_cn.md
```

Run it directly and the model file will be downloaded automatically.

```bash
$ python3 py/ultralytics_YOLO_Detect_YUV420SP.py 
```

If you want to replace other models or use other pictures, you can modify the parameters in the script file.
```bash
$ python3 py/ultralytics_YOLO_Detect_YUV420SP.py -h

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Path to BPU Quantized *.bin Model. RDK X3(Module): Bernoulli2. RDK Ultra: Bayes. RDK X5(Module): Bayes-e. RDK S100: Nash-e. RDK S100P: Nash-m.
  --test-img TEST_IMG   Path to Load Test Image.
  --img-save-path IMG_SAVE_PATH
                        Path to Load Test Image.
  --classes-num CLASSES_NUM
                        Classes Num to Detect.
  --nms-thres NMS_THRES
                        IoU threshold.
  --score-thres SCORE_THRES
                        confidence threshold.
  --reg REG             DFL reg layer.
```




## BenchMark - Performance

### RDK S100P

| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|  
| YOLOv13n  | 640×640 | 80 | 2.8 ms / 353.5 FPS (1 thread  ) <br/> 3.9 ms / 509.0 FPS (2 threads) | 2 ms |  2.5  M  |  6.4   B |  
| YOLOv13s  | 640×640 | 80 | 4.3 ms / 231.7 FPS (1 thread  ) <br/> 7.1 ms / 278.5 FPS (2 threads) | 2 ms |  9.0  M  |  20.8  B |  
| YOLOv13l  | 640×640 | 80 | 12.1 ms / 82.5 FPS (1 thread  ) <br/> 22.7 ms / 87.7 FPS (2 threads) | 2 ms |  27.6 M  |  88.4  B |  
| YOLOv13x  | 640×640 | 80 | 19.7 ms / 50.7 FPS (1 thread  ) <br/> 37.8 ms / 52.7 FPS (2 threads) | 2 ms |  64.0 M  |  199.2 B |  

### RDK S100

| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv13n  | 640×640 | 80 | 3.8 ms / 262.0 FPS (1 thread  ) <br/> 5.2 ms / 378.3 FPS (2 threads) | 2 ms |  2.5  M  |  6.4   B |  
| YOLOv13s  | 640×640 | 80 | 5.8 ms / 169.5 FPS (1 thread  ) <br/> 9.7 ms / 204.9 FPS (2 threads) | 2 ms |  9.0  M  |  20.8  B |  
| YOLOv13l  | 640×640 | 80 | 16.6 ms / 59.8 FPS (1 thread  ) <br/> 31.1 ms / 63.9 FPS (2 threads) | 2 ms |  27.6 M  |  88.4  B |  
| YOLOv13x  | 640×640 | 80 | 26.9 ms / 37.1 FPS (1 thread  ) <br/> 51.6 ms / 38.6 FPS (2 threads) | 2 ms |  64.0 M  |  199.2 B |     

### Performance Test Instructions
1. The performance data tested here are all for models with YUV420SP (nv12) input. There is no significant difference in performance data for models with NCHWRGB input.
2. BPU Latency and BPU Throughput.
- Single-thread latency refers to the delay of processing a single frame by a single thread on a single BPU core, representing the ideal scenario for BPU inference task.
- Multi-thread frame rate means multiple threads simultaneously send tasks to the BPU, where each BPU core can handle tasks from multiple threads. Generally, in engineering projects, controlling the frame delay to be relatively small with 4 threads while fully utilizing the BPU up to 100% can achieve a good balance between throughput (FPS) and frame latency. The BPU of S100/S100P performs quite well, usually requiring only 2 threads to fully utilize the BPU, achieving outstanding frame latency and throughput.
- Data recorded in the table typically reaches a point where throughput does not significantly increase with the number of threads.
- BPU latency and BPU throughput were tested on the board using the following commands:
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin

python3 ../../../resource/tools/batch_perf/batch_perf.py --max 3 --file source/reference_hbm_models/
```

3. The test boards were in their optimal state.
- Optimal state for S100P: CPU consists of 6 × A78AE @ 2.0GHz with full-core Performance scheduling, BPU is 1 × Nash-m @ 1.5GHz, delivering 128TOPS at int8.
- Optimal state for S100: CPU consists of 6 × A78AE @ 1.5GHz with full-core Performance scheduling, BPU is 1 × Nash-e @ 1.0GHz, delivering 80TOPS at int8.

```bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"
```

## Benchmark - Accuracy

### RDK S100 / RDK S100P
Object Detection (COCO2017)
| Model | Pytorch | YUV420SP<br/>Python | YUV420SP<br/>C/C++ | NCHWRGB<br/>C/C++ |
|---------|---------|-------|---------|---------|
| YOLOv13n  | 0.342 | 0.319 (93.27%) | (%) | (%) |
| YOLOv13s  | 0.402 | 0.381 (94.78%) | (%) | (%) |
| YOLOv13l  | 0.458 | 0.443 (96.73%) | (%) | (%) |
| YOLOv13x  | 0.473 | 0.458 (96.83%) | (%) | (%) |


### Accuracy Test Instructions

1. All accuracy data was calculated using Microsoft's unmodified `pycocotools` library, focusing on `Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]`.
2. All test data used the COCO2017 dataset's validation set of 5000 images, inferred directly on-device, saved as JSON files, and processed through third-party testing tools (`pycocotools`) with score thresholds set at 0.25 and NMS thresholds at 0.7.
3. Lower accuracy from `pycocotools` compared to Ultralytics' calculations is normal due to differences in area calculation methods. Our focus is on evaluating quantization-induced precision loss using consistent calculation methods.
4. Some accuracy loss occurs when converting NCHW-RGB888 input to YUV420SP(nv12) input for BPU models, mainly due to color space conversion. Incorporating this during training can mitigate such losses.
5. Slight discrepancies between Python and C/C++ interface accuracies arise from different handling of floating-point numbers during memcpy and conversions.
6. Test scripts can be found in the RDK Model Zoo eval section: [RDK Model Zoo Eval](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools)
7. This table reflects PTQ results using 50 images for calibration and compilation, simulating typical developer scenarios without fine-tuning or QAT, suitable for general validation needs but not indicative of maximum accuracy.


## 进阶开发

### 环境、项目准备

Note: For any errors such as "No such file or directory", "No module named 'xxx'", "command not found", etc., please check carefully. Do not simply copy and run each command one by one. If you do not understand the modification process, please visit the developer community to start learning from YOLOv5.

- Download the iMoonLab/yolov13 repository and configure the environment according to the official Ultralytics documentation.
```bash
git clone https://github.com/iMoonLab/yolov13.git
```
 - Enter the local repository and download the official pre-trained weights.
```bash
cd yolov13
wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
```

### Model Training

- For model training, please refer to the official ultralytics documentation, which is maintained by ultralytics and of very high quality. There are also a great many reference materials on the Internet, and it is not difficult to obtain a model with pre-trained weights like the official one.

- Please note that during training, there is no need to modify any program or the forward method.

Ultralytics YOLO official documentation: https://docs.ultralytics.com/modes/train/


### Export to ONNX

- Uninstall yolo-related command-line commands so that modifications directly in the `./ultralytics/ultralytics` directory take effect.

```bash
$ conda list | grep ultralytics
$ pip list | grep ultralytics # or
# If exists, uninstall
$ conda uninstall ultralytics 
$ pip uninstall ultralytics   # or
```

If it's not straightforward, you can confirm the location of the `ultralytics` directory that needs to be modified using the following Python command:

```bash
>>> import ultralytics
>>> ultralytics.__path__
['/home/wuchao/miniconda3/envs/yolo/lib/python3.11/site-packages/ultralytics']
# or
['/home/wuchao/YOLO11/ultralytics_v11/ultralytics']
```

- Modify the Detect output head to separately output Bounding Box information and Classify information for three feature layers, resulting in a total of six output heads.

File path: `./ultralytics/ultralytics/nn/modules/head.py`, around line 58, replace the `forward` method of the `Detect` class with the following content.

Note: It is suggested to keep the original `forward` method, e.g., rename it to `forward_`, for easier switching back during training.

```python
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result

## If the order of output heads is reversed between bbox and cls, adjust the append order of cv2 and cv3 accordingly,
## then re-export the ONNX model and compile it into a hbm model.

def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
```

- Run the following Python script to export ONNX 

If there is an error saying No module named onnxsim, simply install it. Note, if the generated ONNX model shows too high IR version, set simplify=False. Both settings do not affect the final bin model but improve the readability of the ONNX model in Netron when enabled.

```python
from ultralytics import YOLO
YOLO('yolov11n.pt').export(imgsz=640, format='onnx', simplify=False, opset=19)
```

### Prepare Calibration Data

Refer to the minimalist calibration data preparation script provided in the RDK Model Zoo S: `https://github.com/D-Robotics/rdk_model_zoo_s/blob/s100/resource/tools/generate_calibration_data/generate_cal_data.py` for preparing calibration data.

### Confirm Removal of Dequantization Node Names

Use the Netron visualization tool: `https://netron.app/`

By viewing the ONNX model through Netron, confirm the names of nodes to remove; a handy rule is to remove nodes that contain '64'. Here, 64 = 4 * REG, where REG = 16. Note, the names exported by different versions of Ultralytics may vary, so do not directly apply them.

![](source/imgs/netron_conv_example.jpeg)

See sizes of [1, 80, 80, 64], [1, 40, 40, 64], [1, 20, 20, 64] for their respective names. Corresponding entries should be added to your YAML configuration file.

```yaml
model_parameters:
  onnx_model: 'ultralytcs_YOLO.onnx'
  march: nash-e  # S100: nash-e, S100P: nash-m.
  layer_out_dump: False
  working_dir: 'ultralytcs_YOLO_output'
  output_model_file_prefix: 'ultralytcs_YOLO'
  remove_node_name: "/model.32/cv2.0/cv2.2.2/Conv;/model.32/cv2.1/cv2.1.2/Conv;/model.32/cv2.2/cv2.2.2/Conv;"
```

### Model Compilation

```bash
(bpu_docker) $ hb_compile --config config.yaml
```

### Exception Handling

If the model outputs differ from the reference models in the Model Zoo, this could be due to incorrect node names removed. You can confirm this by checking the bc model information.

```bash
# Quickly generate a bc model
hb_compile --fast-perf --march nash-e --skip compile --model yolo11n.onnx
# View bc model output node information
hb_model_info yolo11n_quantized_model.bc

```
This provides information about the nodes and helps in debugging discrepancies between your model and the reference models provided in the Model Zoo.



### Model compile

```bash
(bpu_docker) $ hb_compile --config config.yaml
```

### Exception Handling

If the output of the Model is inconsistent with the reference model of Model Zoo, the reason might be that the names of the removed nodes are incorrect. This can be confirmed by checking the information of the bc model.

```bash
# Generate a bc model quickly
hb_compile --fast-perf --march nash-e --skip compile --model yolov13n.onnx
# View the output node information of the bc model
hb_model_info yolov13n_quantized_model.bc
```

The following information can be accessed

```bash
2025-06-24 03:17:30,044 INFO ############# Removable node info #############
2025-06-24 03:17:30,044 INFO Node Name                    Node Type
2025-06-24 03:17:30,045 INFO ---------------------------- ----------
2025-06-24 03:17:30,045 INFO /model.32/cv3.0/cv3.0.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv2.0/cv2.0.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv3.1/cv3.1.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv2.1/cv2.1.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv3.2/cv3.2.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv2.2/cv2.2.2/Conv Dequantize
```

Model Zoo provides compilation logs, bc Model information logs and hbm model logs for comparing the differences between the models you obtain yourself and the reference models of Model Zoo.

```bash
./samples/Vision/YOLOv13_iMoonLab/source/reference_logs/
|-- hb_compile_yolov13.txt
|-- hb_model_info_yolov13.txt
`-- hrt_model_exec_model_info_yolov13.txt
```

## References

[ultralytics docs](https://docs.ultralytics.com/)

