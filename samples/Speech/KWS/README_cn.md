[English](./README.md) | 简体中文

# 算法背景

PaddlePaddle（飞桨） 是百度自主研发的、功能强大的 开源深度学习平台，全名是 PArallel Distributed Deep LEarning。它支持丰富的深度学习模型开发、训练与部署，具备如下特点:

1. 支持大规模分布式训练
2. 提供动态图与静态图双模式
3. 适用于工业级应用，广泛应用于语音、图像、自然语言处理等场景
4. 拥有完整的生态，如：PaddleOCR、PaddleSpeech、PaddleNLP 等

PaddleAudio 是飞桨生态中的一个 音频处理与建模库，是 PaddleSpeech 的一部分，主要用于音频任务的预处理、特征提取和建模。其功能包括音频加载与增强（如噪声、混响）、特征提取（如梅尔频率倒谱系数 MFCC、Log-Mel）、音频分类、关键词识别、说话人识别等模型支持、便于音频数据预处理流水线搭建。

PaddleAudio搭载一个语音唤醒算法（KWS）名为MDTC。MDTC（Multi-Scale Dynamic Temporal Convolution） 是一种 多尺度动态时序卷积网络，主要应用于 关键词识别（KWS，Keyword Spotting） 场景。其关键思想：

1. 使用多尺度卷积捕捉不同时间尺度下的音频特征
2. 动态卷积机制增强对关键词变体的鲁棒性
3. 能够在低计算资源下保持高精度，适合部署在边缘设备（如手机、IoT 设备）上

相比传统 CNN 或 RNN 架构，MDTC 更高效且具有更强的时序建模能力，是 PaddleSpeech KWS 模型中的重要算法之一。

# 快速体验

## 环境依赖

在我们拥有一块RDK S100后，我们登录到其环境中。

我们首先克隆代码仓：

```
git clone https://github.com/D-Robotics/rdk_model_zoo_s.git
```

我们随后切换到如下目录：

```
cd samples\Speech\KWS\python
```

安装相关依赖：

```
pip install -r requirements.txt
```

## 效果验证

完成相关依赖安装后，我们看到当前目录下有一个`sample.wav`，这个.wav音频文件包含了唤醒关键词：`hey snips`

我们对该.wav音频文件进行验证。

```
python main.py
```

我们看到如下回显：

![](source/data/image1.jpg)

最后显示的数字，即这段音频为关键词的置信度。

在当前音频下，包含关键词`hey snips`的置信度为~98.5%。

## 速度验证

我们使用如下命令对地瓜异构模型.hbm进行速度的验证

```
hrt_model_exec perf --model_file /root/kws/kws.hbm --frame_count 100
```

我们得到如下结果

* **Frames**: 100
* **Average Latency**: 1.176ms
* **FPS**: 830.875