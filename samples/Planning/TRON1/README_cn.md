[English](./README.md) | 简体中文

# 1. 简介

本仓库的样例为，基于RDK S100的逐际动力点足运控模型部署全流程。

这些年，人形机器人和机器狗频频出圈，成为热门话题。而对于这类机器人而言，运动控制无疑是核心技术之一。强化学习作为当前运动控制的主要方法之一，恰好也能充分发挥 RDK 系列板卡在模型推理方面的优势。因此，我们尝试在 RDK S100 上部署强化学习模型，并探索其在实际机器人平台上的应用。

本次实验机器人采用逐际动力的点足机器人，推理设备采用RDK S100（MLP模型小，资源占用极少），该项目基于逐际动力参与发表过的论文《CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion》，并选择复现和魔改逐际动力提供的Github项目rl-deploy-with-python，我们在该Github项目中加入RDK S100的BPU推理部分。

该项目提供了完整的工程实现，适合作为复现与优化的基础，因此我们最终选择了该方案进行部署与测试。整体流程分为三步：模型训练---->模型量化---->板端部署

# 2. 快速体验

快速体验部分，请跳转RDK S100 nodehub：https://developer.d-robotics.cc/forumDetail/289424806557116663