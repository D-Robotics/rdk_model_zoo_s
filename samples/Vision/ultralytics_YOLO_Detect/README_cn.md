[English](./README.md) | 简体中文

# Ultralytics YOLO Detect

## Abstract
```bash
D-Robotics OpenExplore Version: >= 3.0.31
Ultralytics YOLO Version: >= 8.3.0
```

## Support Models: 

```bash
- YOLOv5u - Detect
- YOLOv8 - Detect
- YOLO11 - Detect
- YOLO12 - Detect
```

## YOLO介绍

![](source/imgs/ultralytics_yolo_detect_performance_comparison.png)


YOLO(You Only Look Once)是一种流行的物体检测和图像分割模型,由华盛顿大学的约瑟夫-雷德蒙(Joseph Redmon)和阿里-法哈迪(Ali Farhadi)开发。YOLO 于 2015 年推出,因其高速度和高精确度而迅速受到欢迎。


 - 2016 年发布的YOLOv2 通过纳入批量归一化、锚框和维度集群改进了原始模型。
2018 年推出的YOLOv3 使用更高效的骨干网络、多锚和空间金字塔池进一步增强了模型的性能。
 - YOLOv4于 2020 年发布, 引入了 Mosaic 数据增强、新的无锚检测头和新的损失函数等创新技术。
 - YOLOv5进一步提高了模型的性能, 并增加了超参数优化、集成实验跟踪和自动导出为常用导出格式等新功能。
 - YOLOv6于 2022 年由美团开源, 目前已用于该公司的许多自主配送机器人。
 - YOLOv7增加了额外的任务, 如 COCO 关键点数据集的姿势估计。
 - YOLOv8是YOLO 的最新版本, 由Ultralytics 提供。YOLOv8 YOLOv8 支持全方位的视觉 AI 任务, 包括检测、分割、姿态估计、跟踪和分类。这种多功能性使用户能够在各种应用和领域中利用YOLOv8 的功能。
 - YOLOv9 引入了可编程梯度信息(PGI) 和广义高效层聚合网络(GELAN)等创新方法。
 - YOLOv10是由清华大学的研究人员使用Ultralytics Python 软件包创建的。该版本通过引入端到端头(End-to-End head),消除了非最大抑制(NMS)要求, 实现了实时目标检测的进步。
 - YOLO11 NEW 🚀: Ultralytics的最新YOLO模型在多个任务上实现了最先进的（SOTA）性能。
 - YOLO12构建以注意力为核心的YOLO框架, 通过创新方法和架构改进, 打破CNN模型在YOLO系列中的主导地位, 实现具有快速推理速度和更高检测精度的实时目标检测。


## 快速体验

```bash
# Make Sure your are in this file
$ cd samples/Vision/ultralytics_YOLO_Detect

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

直接运行, 会自动下载模型文件.

```bash
$ python3 py/ultralytics_YOLO_Detect_YUV420SP.py 
```

如果您想替换其他的模型, 或者使用其他的图片, 可以修改脚本文件内的参数.
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


## 结果分析

![](source/imgs/ultralytics_YOLO_Detect_demo.jpg)

程序自动下载 YOLO12n - Detect 的 BPU HBM 模型, 并完成了对图片的目标检测任务, 可视化结果保存在当前目录下的'py_result.jpg'文件.




## BenchMark - Performance

### RDK S100P

| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv5nu | 640×640 | 80 | 1.5 ms / 650.9 FPS (1 thread  ) <br/> 1.7 ms / 1097.7 FPS (2 threads) <br/> 2.3 ms / 1240.9 FPS (3 threads) | 2 ms |  2.6  M  |  7.7   B |  
| YOLOv5su | 640×640 | 80 | 2.1 ms / 461.0 FPS (1 thread  ) <br/> 2.7 ms / 709.4 FPS (2 threads)   | 2 ms |  9.1  M  |  24.0  B |  
| YOLOv5mu | 640×640 | 80 | 3.6 ms / 275.7 FPS (1 thread  ) <br/> 5.7 ms / 346.6 FPS (2 threads)   | 2 ms |  25.1 M  |  64.2  B |  
| YOLOv5lu | 640×640 | 80 | 6.5 ms / 151.2 FPS (1 thread  ) <br/> 11.6 ms / 170.3 FPS (2 threads)  | 2 ms |  53.2 M  |  135.0 B |  
| YOLOv5xu | 640×640 | 80 | 11.6 ms / 85.9 FPS (1 thread  ) <br/> 21.5 ms / 92.4 FPS (2 threads)   | 2 ms |  97.2 M  |  246.4 B |  
| YOLOv8n  | 640×640 | 80 | 1.6 ms / 612.8 FPS (1 thread  ) <br/> 1.8 ms / 1047.2 FPS (2 threads)  | 2 ms |  3.2  M  |  8.7   B |  
| YOLOv8s  | 640×640 | 80 | 2.3 ms / 414.6 FPS (1 thread  ) <br/> 3.2 ms / 599.2 FPS (2 threads)   | 2 ms |  11.2 M  |  28.6  B |  
| YOLOv8m  | 640×640 | 80 | 4.1 ms / 238.6 FPS (1 thread  ) <br/> 6.8 ms / 291.0 FPS (2 threads)   | 2 ms |  25.9 M  |  78.9  B |  
| YOLOv8l  | 640×640 | 80 | 7.8 ms / 127.9 FPS (1 thread  ) <br/> 14.0 ms / 141.4 FPS (2 threads)  | 2 ms |  43.7 M  |  165.2 B |  
| YOLOv8x  | 640×640 | 80 | 11.7 ms / 85.0 FPS (1 thread  ) <br/> 21.7 ms / 91.5 FPS (2 threads)   | 2 ms |  68.2 M  |  257.8 B |  
| YOLO11n  | 640×640 | 80 | 1.6 ms / 585.4 FPS (1 thread  ) <br/> 1.9 ms  / 1028.4 FPS (2 threads) | 2 ms |  2.6  M  |  6.5   B |  
| YOLO11s  | 640×640 | 80 | 2.3 ms / 417.0 FPS (1 thread  ) <br/> 3.2 ms  / 603.2 FPS (2 threads)  | 2 ms |  9.4  M  |  21.5  B |  
| YOLO11m  | 640×640 | 80 | 4.4 ms / 221.6 FPS (1 thread  ) <br/> 7.4 ms  / 266.0 FPS (2 threads)  | 2 ms |  20.1 M  |  68.0  B |  
| YOLO11l  | 640×640 | 80 | 5.5 ms / 178.4 FPS (1 thread  ) <br/> 9.6 ms  / 206.1 FPS (2 threads)  | 2 ms |  25.3 M  |  86.9  B |  
| YOLO11x  | 640×640 | 80 | 9.8 ms / 100.9 FPS (1 thread  ) <br/> 18.1 ms / 109.6 FPS (2 threads)  | 2 ms |  56.9 M  |  194.9 B |  
| YOLO12n  | 640×640 | 80 | 2.5 ms / 395.4 FPS (1 thread  ) <br/> 3.5 ms / 554.0 FPS (2 threads)   | 2 ms |  2.6  M  |  6.5   B |  
| YOLO12s  | 640×640 | 80 | 4.0 ms / 247.8 FPS (1 thread  ) <br/> 6.5 ms / 304.6 FPS (2 threads)   | 2 ms |  9.3  M  |  21.4  B |  
| YOLO12m  | 640×640 | 80 | 7.1 ms / 139.5 FPS (1 thread  ) <br/> 12.7 ms / 155.8 FPS (2 threads)  | 2 ms |  20.2 M  |  67.5  B |  
| YOLO12l  | 640×640 | 80 | 11.2 ms / 88.4 FPS (1 thread  ) <br/> 20.9 ms / 95.0 FPS (2 threads)   | 2 ms |  26.4 M  |  88.9  B |  
| YOLO12x  | 640×640 | 80 | 18.9 ms / 52.7 FPS (1 thread  ) <br/> 36.2 ms / 55.0 FPS (2 threads)   | 2 ms |  59.1 M  |  199.0 B |  



### RDK S100

| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv5nu | 640×640 | 80 | 2.0 ms / 487.0 FPS (1 thread  ) <br/> 2.2 ms / 864.5 FPS (2 threads) <br/> 3.2 ms / 896.6 FPS (3 threads) | 2 ms |  2.6  M  |  7.7   B |  
| YOLOv5su | 640×640 | 80 | 2.9 ms / 340.5 FPS (1 thread  ) <br/> 3.9 ms / 498.2 FPS (2 threads)   | 2 ms |  9.1  M  |  24.0  B |  
| YOLOv5mu | 640×640 | 80 | 5.0 ms / 196.2 FPS (1 thread  ) <br/> 8.2 ms / 239.9 FPS (2 threads)   | 2 ms |  25.1 M  |  64.2  B |  
| YOLOv5lu | 640×640 | 80 | 9.5 ms / 104.0 FPS (1 thread  ) <br/> 17.2 ms / 115.5 FPS (2 threads)  | 2 ms |  53.2 M  |  135.0 B |  
| YOLOv5xu | 640×640 | 80 | 16.9 ms / 58.9 FPS (1 thread  ) <br/> 31.8 ms / 62.6 FPS (2 threads)   | 2 ms |  97.2 M  |  246.4 B |  
| YOLOv8n  | 640×640 | 80 | 2.0 ms / 485.1 FPS (1 thread  ) <br/> 2.4 ms / 798.0 FPS (2 threads)   | 2 ms |  3.2  M  |  8.7   B |  
| YOLOv8s  | 640×640 | 80 | 3.1 ms / 312.7 FPS (1 thread  ) <br/> 4.7 ms / 416.7 FPS (2 threads)   | 2 ms |  11.2 M  |  28.6  B |  
| YOLOv8m  | 640×640 | 80 | 5.8 ms / 170.0 FPS (1 thread  ) <br/> 10.0 ms / 198.3 FPS (2 threads)  | 2 ms |  25.9 M  |  78.9  B |  
| YOLOv8l  | 640×640 | 80 | 11.1 ms / 89.1 FPS (1 thread  ) <br/> 20.4 ms / 97.3 FPS (2 threads)   | 2 ms |  43.7 M  |  165.2 B |  
| YOLOv8x  | 640×640 | 80 | 17.0 ms / 58.6 FPS (1 thread  ) <br/> 31.9 ms / 62.3 FPS (2 threads)   | 2 ms |  68.2 M  |  257.8 B |  
| YOLO11n  | 640×640 | 80 | 2.1 ms / 466.6 FPS (1 thread  ) <br/> 2.6 ms / 741.0 FPS (2 threads)   | 2 ms |  2.6  M  |  6.5   B |  
| YOLO11s  | 640×640 | 80 | 3.1 ms / 313.9 FPS (1 thread  ) <br/> 4.7 ms / 419.8 FPS (2 threads)   | 2 ms |  9.4  M  |  21.5  B |  
| YOLO11m  | 640×640 | 80 | 6.3 ms / 157.3 FPS (1 thread  ) <br/> 10.9 ms / 181.9 FPS (2 threads)  | 2 ms |  20.1 M  |  68.0  B |  
| YOLO11l  | 640×640 | 80 | 7.9 ms / 125.8 FPS (1 thread  ) <br/> 14.0 ms / 141.5 FPS (2 threads)  | 2 ms |  25.3 M  |  86.9  B |  
| YOLO11x  | 640×640 | 80 | 14.1 ms / 70.3 FPS (1 thread  ) <br/> 26.4 ms / 75.4 FPS (2 threads)   | 2 ms |  56.9 M  |  194.9 B |  
| YOLO12n  | 640×640 | 80 | 3.3 ms / 293.3 FPS (1 thread  ) <br/> 5.2 ms / 382.1 FPS (2 threads)   | 2 ms |  2.6  M  |  6.5   B |  
| YOLO12s  | 640×640 | 80 | 5.6 ms / 174.7 FPS (1 thread  ) <br/> 9.7 ms / 204.7 FPS (2 threads)   | 2 ms |  9.3  M  |  21.4  B |  
| YOLO12m  | 640×640 | 80 | 10.4 ms / 95.7 FPS (1 thread  ) <br/> 18.9 ms / 104.8 FPS (2 threads)  | 2 ms |  20.2 M  |  67.5  B |  
| YOLO12l  | 640×640 | 80 | 16.6 ms / 60.1 FPS (1 thread  ) <br/> 31.2 ms / 63.8 FPS (2 threads)   | 2 ms |  26.4 M  |  88.9  B |  
| YOLO12x  | 640×640 | 80 | 27.6 ms / 36.1 FPS (1 thread  ) <br/> 53.2 ms / 37.4 FPS (2 threads)   | 2 ms |  59.1 M  |  199.0 B |  



### Performance Test Instructions
1. 此处测试的均为YUV420SP (nv12) 输入的模型的性能数据. NCHWRGB输入的模型的性能数据与其无明显差距.
2. BPU延迟与BPU吞吐量。
 - 单线程延迟为单帧,单线程,单BPU核心的延迟,BPU推理一个任务最理想的情况。
 - 多线程帧率为多个线程同时向BPU塞任务, 每个BPU核心可以处理多个线程的任务, 一般工程中4个线程可以控制单帧延迟较小,同时吃满所有BPU到100%,在吞吐量(FPS)和帧延迟间得到一个较好的平衡。S100 / S100P的BPU整体比较厉害, 一般2个线程就可以将BPU吃满, 帧延迟和吞吐量都非常出色。
 - 表格中一般记录到吞吐量不再随线程数明显增加的数据。
 - BPU延迟和BPU吞吐量使用以下命令在板端测试
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin

python3 ../../../resource/tools/batch_perf/batch_perf.py --max 3 --file source/reference_hbm_models/
```
3. 测试板卡为最佳状态。

 - S100P的状态为最佳状态: CPU为6 × A78AE @ 2.0GHz, 全核心Performance调度, BPU为1 × Nash-m @ 1.5GHz, 128TOPS @ int8.
 - S100的状态为最佳状态: CPU为6 × A78AE @ 1.5GHz, 全核心Performance调度, BPU为1 × Nash-e @ 1.0GHz, 80TOPS @ int8.

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
| YOLOv5nu | 0.275 | 0.259 (94.18%) | (%) | (%) |
| YOLOv5su | 0.362 | 0.349 (96.41%) | (%) | (%) |
| YOLOv5mu | 0.417 | 0.400 (95.92%) | (%) | (%) |
| YOLOv5lu | 0.449 | 0.436 (97.10%) | (%) | (%) |
| YOLOv5xu | 0.458 | 0.440 (96.07%) | (%) | (%) |
| YOLOv8n  | 0.306 | 0.291 (95.10%) | (%) | (%) |
| YOLOv8s  | 0.384 | 0.368 (95.83%) | (%) | (%) |
| YOLOv8m  | 0.433 | 0.417 (96.30%) | (%) | (%) |
| YOLOv8l  | 0.454 | 0.437 (96.26%) | (%) | (%) |
| YOLOv8x  | 0.465 | 0.446 (95.91%) | (%) | (%) |
| YOLO11n  | 0.323 | 0.303 (93.81%) | (%) | (%) |
| YOLO11s  | 0.394 | 0.375 (95.18%) | (%) | (%) |
| YOLO11m  | 0.437 | 0.416 (95.19%) | (%) | (%) |
| YOLO11l  | 0.452 | 0.429 (94.91%) | (%) | (%) |
| YOLO11x  | 0.466 | 0.442 (94.85%) | (%) | (%) |
| YOLO12n  | 0.334 | 0.311 (93.11%) | (%) | (%) |
| YOLO12s  | 0.397 | 0.381 (95.97%) | (%) | (%) |
| YOLO12m  | 0.444 | 0.421 (94.82%) | (%) | (%) |
| YOLO12l  | 0.454 | 0.430 (94.71%) | (%) | (%) |
| YOLO12x  | 0.466 | 0.439 (94.21%) | (%) | (%) |

### Accuracy Test Instructions

1. 所有的精度数据使用微软官方的无修改的`pycocotools`库进行计算, 取的精度标准为`Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]`的数据。
2. 所有的测试数据均使用`COCO2017`数据集的val验证集的5000张照片, 在板端直接推理, dump保存为json文件, 送入第三方测试工具`pycocotools`库进行计算, 分数的阈值为0.25, nms的阈值为0.7。
3. pycocotools计算的精度比ultralytics计算的精度会低一些是正常现象, 主要原因是pycocotools是取矩形面积, ultralytics是取梯形面积, 我们主要是关注同样的一套计算方式去测试定点模型和浮点模型的精度, 从而来评估量化过程中的精度损失. 
4. BPU模型在量化NCHW-RGB888输入转换为YUV420SP(nv12)输入后, 也会有一部分精度损失, 这是由于色彩空间转化导致的, 在训练时加入这种色彩空间转化的损失可以避免这种精度损失。
5. Python接口和C/C++接口的精度结果有细微差异, 主要在于Python和C/C++的一些数据结构进行memcpy和转化的过程中, 对浮点数的处理方式不同, 导致的细微差异.
6. 测试脚本请参考RDK Model Zoo的eval部分: https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools
7. 本表格是使用PTQ(训练后量化)使用50张图片进行校准和编译的结果, 用于模拟普通开发者第一次直接编译的精度情况, 并没有进行精度调优或者QAT(量化感知训练), 满足常规使用验证需求, 不代表精度上限.


## 进阶开发

### 高性能计算流程介绍

![](source/imgs/ultralytics_YOLO_Detect_DataFlow.png)

公版处理流程中, 是会对8400个bbox完全计算分数, 类别和xyxy坐标, 这样才能根据GT去计算损失函数。但是我们在部署中, 只需要合格的bbox就好了, 并不需要对8400个bbox完全计算。
优化处理流程中, 主要就是利用Sigmoid函数单调性做到了先筛选, 再计算。同时利用Python的numpy的高级索引, 对DFL和特征解码的部分也做到了先筛选, 再计算, 节约了大量的计算, 从而后处理在CPU上, 利用numpy, 可以做到单核单帧单线程5毫秒。

 - Classify部分,Dequantize操作
在模型编译时,如果选择了移除所有的反量化算子,这里需要在后处理中手动对Classify部分的三个输出头进行反量化在. 查看反量化系数的方式有多种, 可以查看`hb_combine`时产物的日志, 也可通过BPU推理接口的API来获取。
注意,这里每一个C维度的反量化系数都是不同的,每个头都有80个反量化系数,可以使用numpy的广播直接乘。
此处反量化在bin模型中实现,所以拿到的输出是float32的。

 - Classify部分,ReduceMax操作
ReduceMax操作是沿着Tensor的某一个维度找到最大值,此操作用于找到8400个Grid Cell的80个分数的最大值。操作对象是每个Grid Cell的80类别的值,在C维度操作。注意,这步操作给出的是最大值,并不是80个值中最大值的索引。
激活函数Sigmoid具有单调性,所以Sigmoid作用前的80个分数的大小关系和Sigmoid作用后的80个分数的大小关系不会改变。
$$Sigmoid(x)=\frac{1}{1+e^{-x}}$$
$$Sigmoid(x_1) > Sigmoid(x_2) \Leftrightarrow x_1 > x_2$$
综上,bin模型直接输出的最大值(反量化完成)的位置就是最终分数最大值的位置,bin模型输出的最大值经过Sigmoid计算后就是原来onnx模型的最大值。

 - Classify部分,Threshold（TopK）操作
此操作用于找到8400个Grid Cell中,符合要求的Grid Cell。操作对象为8400个Grid Cell,在H和W的维度操作。如果您有阅读我的程序,你会发现我将后面H和W维度拉平了,这样只是为了程序设计和书面表达的方便,它们并没有本质上的不同。
我们假设某一个Grid Cell的某一个类别的分数记为$x$,激活函数作用完的整型数据为$y$,阈值筛选的过程会给定一个阈值,记为$C$,那么此分数合格的**充分必要条件**为: 

$$y=Sigmoid(x)=\frac{1}{1+e^{-x}}>C$$

由此可以得出此分数合格的**充分必要条件**为: 

$$x > -ln\left(\frac{1}{C}-1\right)$$

此操作会符合条件的Grid Cell的索引（indices）和对应Grid Cell的最大值,这个最大值经过Sigmoid计算后就是这个Grid Cell对应类别的分数了。

 - Classify部分,GatherElements操作和ArgMax操作
使用Threshold(TopK)操作得到的符合条件的Grid Cell的索引(indices),在GatherElements操作中获得符合条件的Grid Cell,使用ArgMax操作得到具体是80个类别中哪一个最大,得到这个符合条件的Grid Cell的类别。

 - Bounding Box部分,GatherElements操作和Dequantize操作
使用Threshold(TopK)操作得到的符合条件的Grid Cell的索引(indices),在GatherElements操作中获得符合条件的Grid Cell,这里每一个C维度的反量化系数都是不同的,每个头都有64个反量化系数,可以使用numpy的广播直接乘,得到1×64×k×1的bbox信息。

 - Bounding Box部分,DFL: SoftMax+Conv操作
每一个Grid Cell会有4个数字来确定这个框框的位置,DFL结构会对每个框的某条边基于anchor的位置给出16个估计,对16个估计求SoftMax,然后通过一个卷积操作来求期望,这也是Anchor Free的核心设计,即每个Grid Cell仅仅负责预测1个Bounding box。假设在对某一条边偏移量的预测中,这16个数字为 $ l_p $ 或者$(t_p, t_p, b_p)$,其中$p = 0,1,...,15$那么偏移量的计算公式为: 

$$\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{S}}, S =\sum_{p=0}^{15}{e^{l_p}}$$

 - Bounding Box部分,Decode: dist2bbox(ltrb2xyxy)操作
此操作将每个Bounding Box的ltrb描述解码为xyxy描述,ltrb分别表示左上右下四条边距离相对于Grid Cell中心的距离,相对位置还原成绝对位置后,再乘以对应特征层的采样倍数,即可还原成xyxy坐标,xyxy表示Bounding Box的左上角和右下角两个点坐标的预测值。
![](imgs/ltrb2xyxy.jpg)

图片输入为$Size=640$,对于Bounding box预测分支的第$i$个特征图$(i=1, 2, 3)$,对应的下采样倍数记为$Stride(i)$,在YOLOv8 - Detect中,$Stride(1)=8, Stride(2)=16, Stride(3)=32$,对应特征图的尺寸记为$n_i = {Size}/{Stride(i)}$,即尺寸为$n_1 = 80, n_2 = 40 ,n_3 = 20$三个特征图,一共有$n_1^2+n_2^2+n_3^3=8400$个Grid Cell,负责预测8400个Bounding Box。
对特征图i,第x行y列负责预测对应尺度Bounding Box的检测框,其中$x,y \in [0, n_i)\bigcap{Z}$,$Z$为整数的集合。DFL结构后的Bounding Box检测框描述为$ltrb$描述,而我们需要的是$xyxy$描述,具体的转化关系如下: 

$$x_1 = (x+0.5-l)\times{Stride(i)}$$

$$y_1 = (y+0.5-t)\times{Stride(i)}$$

$$x_2 = (x+0.5+r)\times{Stride(i)}$$

$$y_1 = (y+0.5+b)\times{Stride(i)}$$

最终的检测结果,包括类别(id),分数(score)和位置(xyxy)。

### 环境、项目准备

注: 任何No such file or directory, No module named "xxx", command not found.等报错请仔细检查, 请勿逐条复制运行, 如果对修改过程不理解请前往开发者社区从YOLOv5开始了解。

 - 下载ultralytics/ultralytics仓库, 并参考ultralytics官方文档, 配置好环境.
```bash
git clone https://github.com/ultralytics/ultralytics.git
```
 - 进入本地仓库, 下载ultralytics官方的预训练权重, 这里以YOLO11n-Detect模型为例.
```bash
cd ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

### 模型训练

 - 模型训练请参考ultralytics官方文档, 这个文档由ultralytics维护, 质量非常的高。网络上也有非常多的参考材料, 得到一个像官方一样的预训练权重的模型并不困难。
 - 请注意, 训练时无需修改任何程序, 无需修改forward方法。

Ultralytics YOLO 官方文档: https://docs.ultralytics.com/modes/train/


### 导出为onnx

 - 卸载yolo相关的命令行命令, 这样直接修改`./ultralytics/ultralytics`目录内的内容即可生效.

```bash
$ conda list | grep ultralytics
$ pip list | grep ultralytics # 或者
# 如果存在, 则卸载
$ conda uninstall ultralytics 
$ pip uninstall ultralytics   # 或者
```

如果不是很顺利, 可以通过以下Python命令确认需要修改的`ultralytics`目录的位置.

```bash
>>> import ultralytics
>>> ultralytics.__path__
['/home/wuchao/miniconda3/envs/yolo/lib/python3.11/site-packages/ultralytics']
# 或者
['/home/wuchao/YOLO11/ultralytics_v11/ultralytics']
```

 - 修改Detect的输出头, 直接将三个特征层的Bounding Box信息和Classify信息分开输出, 一共6个输出头。

文件目录: ./ultralytics/ultralytics/nn/modules/head.py, 约第58行, `Detect`类的forward方法替换成以下内容.

注: 建议您保留好原本的`forward`方法, 例如改一个其他的名字`forward_`, 方便在训练的时候换回来。

```python
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result

## 如果输出头顺序刚好是bbox和cls反的, 可以使用如下修改方式, 调换cv2和cv3的append顺序
## 然后再重新导出onnx, 编译为bin模型
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
```

 - 修改优化后的Attntion模块(YOLO11 可选)

文件目录: `ultralytics/nn/modules/block.py`, 约第868行, `Attntion`类的`forward`方法替换成以下内容. 主要的优化点是去除了一些无用的数据搬运操作, 同时将Reduce的维度变为C维度, 对BPU更加友好, 不需要重新训练模型.

注: 建议您保留好原本的`forward`方法,例如改一个其他的名字`forward_`, 方便在训练的时候换回来。

```python
class Attention(nn.Module):
    def forward(self, x):  # original
        pass
    def forward(self, x):  # RDK
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.permute(0, 3, 1, 2).contiguous()  # CHW2HWC like
        max_attn = attn.max(dim=1, keepdim=True).values 
        exp_attn = torch.exp(attn - max_attn)
        sum_attn = exp_attn.sum(dim=1, keepdim=True)
        attn = exp_attn / sum_attn
        attn = attn.permute(0, 2, 3, 1).contiguous()  # HWC2CHW like
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
```

 - 修改优化后的AAttn模块(YOLO12 可选)

文件目录: `ultralytics/nn/modules/block.py`, 约第1159行, `AAttn`类的`forward`方法替换成以下内容. 主要的优化点是去除了一些无用的数据搬运操作，同时将Reduce的维度变为C维度，对BPU更加友好。并且不需要重新训练模型。

注: 建议您保留好原本的`forward`方法,例如改一个其他的名字`forward_`, 方便在训练的时候换回来。

```python
class AAttn(nn.Module):
    def forward(self, x):  # original
        pass
    def forward(self, x):  # RDK
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split(
            [self.head_dim, self.head_dim, self.head_dim], dim=3
        )
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.permute(0, 3, 1, 2).contiguous()  # CHW2HWC like
        max_attn = attn.max(dim=1, keepdim=True).values 
        exp_attn = torch.exp(attn - max_attn)
        sum_attn = exp_attn.sum(dim=1, keepdim=True)
        attn = exp_attn / sum_attn
        attn = attn.permute(0, 2, 3, 1).contiguous()  # HWC2CHW like
        x = (v @ attn.transpose(-2, -1))
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)
        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.pe(v)
        x = self.proj(x)
        return x
```

 - 运行以下Python脚本进行ONNX导出
如果有**No module named onnxsim**报错, 安装一个即可. 注意, 如果生成的onnx模型显示ir版本过高, 可以将simplify=False。两种设置对最终bin模型没有影响, 打开后可以提升onnx模型在netron中的可读性.

```python
from ultralytics import YOLO
YOLO('yolov11n.pt').export(imgsz=640, format='onnx', simplify=False, opset=19)
```

### 准备校准数据

参考RDK Model Zoo S提供的极简的校准数据准备脚本: `samples/Vision/ultralytics_YOLO_Detect/source/generate_cal_data.py `进行校准数据的准备。


### 确认移除反量化节点的名称

Netron可视化工具:` https://netron.app/`

通过Netron查看对ONNX模型进行可视化, 确认需要移除的节点名称, 这里有一个小口诀, 就是带64的都移除. 这里的64 = 4 * REG, REG = 16. 注意, 不同版本的Ultralytics导出的ONNX的名称是不同的, 请勿直接套用.

![](source/imgs/netron_conv_example.jpeg)

看到大小为[1, 80, 80, 64], [1, 40, 40, 64], [1, 20, 20, 64]的三个输出的名称为/model.23/cv2.0/cv2.0.2/Conv, /model.23/cv2.1/cv2.1.2/Conv, /model.23/cv2.2/cv2.2.2/Conv;

对应的yaml中填入对应的名称.

```yaml
model_parameters:
  onnx_model: 'ultralytcs_YOLO.onnx'
  march: nash-e  # S100: nash-e, S100P: nash-m.
  layer_out_dump: False
  working_dir: 'ultralytcs_YOLO_output'
  output_model_file_prefix: 'ultralytcs_YOLO'
  remove_node_name: "/model.23/cv2.0/cv2.0.2/Conv;/model.23/cv2.1/cv2.1.2/Conv;/model.23/cv2.2/cv2.2.2/Conv;"
  # Reference remove_node_name
  # YOLOv5u: /model.24/cv2.0/cv2.0.2/Conv;/model.24/cv2.1/cv2.1.2/Conv;/model.24/cv2.2/cv2.2.2/Conv;
  # YOLOv8: /model.22/cv2.0/cv2.0.2/Conv;/model.22/cv2.1/cv2.1.2/Conv;/model.22/cv2.2/cv2.2.2/Conv;
  # YOLO11: /model.23/cv2.0/cv2.0.2/Conv;/model.23/cv2.1/cv2.1.2/Conv;/model.23/cv2.2/cv2.2.2/Conv;
  # YOLO12: /model.21/cv2.0/cv2.0.2/Conv;/model.21/cv2.1/cv2.1.2/Conv;/model.21/cv2.2/cv2.2.2/Conv;

```

### 模型编译
```bash
(bpu_docker) $ hb_compile --config config.yaml
```

### 异常处理

如果模型的输出情况与Model Zoo参考模型不一致, 原因可能是移除的节点名称错误, 可通过查看bc模型的信息来确认.

```bash
# 快速产生一个bc模型
hb_compile --fast-perf --march nash-e --skip compile --model yolo11n.onnx
# 查看bc模型的输出节点信息
hb_model_info yolo11n_quantized_model.bc
```

可查阅到以下信息

```bash
INFO ############# Removable node info #############
INFO Node Name                                          Node Type
INFO -------------------------------------------------- ----------
INFO /model.23/cv3.0/cv3.0.2/Conv                       Dequantize
INFO /model.23/cv2.0/cv2.0.2/Conv                       Dequantize
INFO /model.23/cv4.0/cv4.0.2/Conv                       Dequantize
INFO /model.23/cv3.1/cv3.1.2/Conv                       Dequantize
INFO /model.23/cv2.1/cv2.1.2/Conv                       Dequantize
INFO /model.23/cv4.1/cv4.1.2/Conv                       Dequantize
INFO /model.23/cv3.2/cv3.2.2/Conv                       Dequantize
INFO /model.23/cv2.2/cv2.2.2/Conv                       Dequantize
INFO /model.23/cv4.2/cv4.2.2/Conv                       Dequantize
INFO /model.23/proto/cv3/act/Mul_output_0_HzCalibration Dequantize
```

Model Zoo提供编译日志, bc模型信息日志和hbm模型日志, 用于比较您自己获得的模型和Model Zoo参考模型的区别.

```bash
./samples/Vision/ultralytics_YOLO_Detect/source/reference_logs/
|-- hb_compile_yolo11n.txt
|-- hb_compile_yolo12n.txt
|-- hb_compile_yolov5nu.txt
|-- hb_compile_yolov8n.txt
|-- hb_model_info_yolo11n.txt
|-- hb_model_info_yolo12n.txt
|-- hb_model_info_yolov5nu.txt
|-- hb_model_info_yolov8n.txt
|-- hrt_model_exec_model_info_yolo11n.txt
|-- hrt_model_exec_model_info_yolo12n.txt
|-- hrt_model_exec_model_info_yolov5nu.txt
`-- hrt_model_exec_model_info_yolov8n.txt
```

## 参考

[ultralytics](https://docs.ultralytics.com/)

[Github: yolo12](https://github.com/sunsmarterjie/yolo12)

[yolo12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524)

