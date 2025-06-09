English| [简体中文](./README_cn.md)

[English](./README.md) | 简体中文

# 1. Introduction

The sample in this repository is the whole process of deploying the foot motion control model of Zhujidongli based on RDK S100. 

In recent years, humanoid robots and robotic dogs have frequently broken into other demographic and become hot topics. For this kind of robots, motion control is undoubtedly one of the core technologies. As one of the main methods of current motion control, reinforcement learning can also give full play to the advantages of RDK series boards in model inference. Therefore, we tried to deploy the reinforcement learning model on the RDK S100 and explore its application on the actual robot platform. 

The robot used in this experiment is the point-foot robot of Zhuji Dynamics, and the inference device is RDK S100 (the MLP model is small and takes up very few resources). This project is based on the paper "CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion" published by Zhuji Dynamics, and we choose to reproduce and modify the Github project rl-deploy-with-python provided by Zhuji Dynamics. We add the BPU inference part of RDK S100 to this Github project. 

The project provides a complete engineering implementation, which is suitable as a basis for reproduction and optimization. Therefore, we finally chose this solution for deployment and testing. The overall process is divided into three steps: Model Training ----> Model Quantization ----> Board Deployment 

# 2. Quick Experience 

For the quick experience section, please jump to RDK S100 nodehub:https://developer.d-robotics.cc/forumDetail/289424806557116663