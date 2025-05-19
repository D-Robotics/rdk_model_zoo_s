[English](./README.md) | 简体中文

# 算法背景

Hugging Face 是一家专注于自然语言处理（NLP）、语音识别、计算机视觉等人工智能领域的开源技术公司，以构建高质量的 AI 工具链和活跃的开源社区而闻名。其核心产品是 Transformers 库，该库汇集了成千上万种预训练模型，如 BERT、GPT、T5、Wav2Vec2 等，支持 PyTorch、TensorFlow 和 JAX 等深度学习框架，并提供统一的 API 和预处理工具，使模型的加载、微调与部署变得简单高效。

其中Hub包括：

1. 开源模型托管平台，允许开发者上传、下载、分享模型。
2. 包含权重、配置文件、预处理脚本等。

Wav2Vec2 是由 Facebook AI Research 提出的端到端语音识别模型，使用自监督学习方法在大量未标注的原始音频上进行预训练，然后在有监督语音数据上进行微调。该模型以原始波形为输入，先通过卷积神经网络提取低级特征，再利用 Transformer 编码上下文信息，并使用掩蔽预测机制提升学习能力。Wav2Vec2 不依赖传统的声学模型和语言模型，显著简化了语音识别系统的架构，同时在如 LibriSpeech 等标准数据集上取得了优异表现，尤其在低资源、多语言场景中展现出强大的迁移能力。目前，HuggingFace 已将 Wav2Vec2 完整集成到其模型库中，用户可通过简单的几行代码加载预训练模型并实现高精度的语音转文字功能，大大降低了语音识别系统的开发门槛。


# 快速体验

## 环境依赖

在我们拥有一块RDK S100后，我们登录到其环境中。

我们首先克隆代码仓：

```
git clone https://github.com/D-Robotics/rdk_model_zoo_s.git
```

我们随后切换到如下目录：

```
cd samples\Speech\ASR\python
```

安装相关依赖：

```
pip install -r requirements.txt
```

下载模型：

```
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/asr/asr.hbm
```

## 效果验证

完成相关依赖安装后，我们看到当前目录下有一个`chi_sound.wav`，这个.wav音频文件包含了一段中文语音。

我们对该.wav音频文件进行验证。

```
python main.py
```

我们看到如下回显：

![](source/data/print.jpg)

以上这段回显的前数行为加载BPU推理库相应的资源。

最后一行为对`chi_sound.wav`这段音频的前三秒进行ASR的结果。

## 速度验证

我们使用如下命令对地瓜异构模型.hbm进行速度的验证

```
hrt_model_exec perf --model_file asr.hbm --frame_count 100
```

我们得到如下结果

![](source/data/perf.jpg)

* **Frames**: 100
* **Average Latency**: 34.426ms
* **FPS**: 29.008