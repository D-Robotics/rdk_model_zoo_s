[English](./README.md) | 简体中文
当然可以！以下是你提供内容的中文 `README.md` 格式文档，适合直接放到项目主页或开源仓库中使用：


# 项目简介

在机器人领域，**弥合仿真环境与现实世界的动力学差异**是实现精准运动控制的关键挑战。地瓜机器人在地平线机器人实验室优化的论文框架《Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills》的基础上，成功在 **RDK S100** 平台上部署了其优化后的 **ASAP 框架**。

优化后的 ASAP 框架采用**两阶段训练策略**，结合 **Delta 动作模型**，实现了人形机器人在真实环境中的高效全身运动控制，有效降低了仿真与现实之间的动力学差异，实机表现稳定、出色。

本次技术突破的亮点在于**大幅降低部署资源消耗**：
- 在 **BPU 上模型推理占用率仅为 2%**；
- 相较纯 CPU 推理，**CPU 占用降低达 250%**；
- 为视觉检测、目标识别、路径导航和智能决策等任务释放更多算力，**全面提升系统综合能力**。

优化后的 ASAP 框架尚未对外开源，**预计将于 6 月份发布**。本仓库当前仅提供 **模型量化流程**，训练及部署流程将在后续更新中补充，欢迎持续关注！

---

# 模型量化说明

## 环境准备

1. 拉取 S100 工具链 Docker 镜像（请联系工程师获取 Docker 镜像和 OE 包）。
2. 解压用于模型量化的文件夹（见文末附件），包含：
   - 配置文件
   - 校准数据

3. 将导出的 ONNX 模型放入该文件夹，**模型名称需与配置文件中一致**。
4. 将该文件夹挂载到 Docker 容器中，并启动容器：

```bash
sudo docker run -it --entrypoint="/bin/bash" \
-v 主机文件夹路径:/g1_model_convert_s100 \
镜像ID
````

## 量化编译流程

```bash
# 进入挂载目录
cd /g1_model_convert_s100

# 进行模型量化，生成 model_output 文件夹，输出 .bin 模型
hb_compile -c dance_bpu.yaml
```
![](https://developer.d-robotics.cc/api/v1/static/imgData/1748580922266.jpg)
不做优化的话，量化精度不够高。需要对量化过程中产生的优化后的float模型（xx_optimized_float_model）进行拆解，拆解脚本见文末。（修改模型名称并将卷积算子名称替换为上图圈红的卷积算子名称）

## 提高精度：float 模型拆解

1. 使用提供的脚本拆解优化后的 float 模型：

   * 修改脚本中的模型名称；
   * 将卷积算子名称替换为配置文件中圈红的算子名称。

```bash
python split_float_model.py
```

2. 使用拆解后的模型重新量化，使用原配置文件：

```bash
hb_compile -c dance_bpu.yaml
```
![](https://developer.d-robotics.cc/api/v1/static/imgData/1748580943990.jpg)
---

# 效果展示

B 站视频演示：[从仿真到实机无缝迁移！地瓜机器人基于RDK S100复现CoRL获奖论文仿生步态](https://www.bilibili.com/video/BV154d2YTEn1/)

---

# 附件下载

> 量化所需文件包（含配置文件与校准数据）：

* **文件名**：`g1_model_convert_s100.zip`
* **百度网盘链接**：[https://pan.baidu.com/s/1DMyqMGk1IcE9tuk9OkM30w?pwd=fkt3](https://pan.baidu.com/s/1DMyqMGk1IcE9tuk9OkM30w?pwd=fkt3)
* **提取码**：`fkt3`

---

# 鸣谢

特别感谢 **地平线机器人实验室** 的大力支持！
感谢 **王煜城**、**王凯辉**、**陈迈越** 三位工程师的鼎力相助！


