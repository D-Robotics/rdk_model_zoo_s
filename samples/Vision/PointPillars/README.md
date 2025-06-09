English| [简体中文](./README_cn.md)

# Algorithm Background

OpenPCDet is an open-source 3D Object Detection framework developed by Southwest Jiaotong University. It focuses on Object Detection tasks based on LiDAR point cloud data and is commonly used in autonomous driving scenarios. The framework has the following features:

1. Modularization design: The network structure is clear, which is convenient for expansion and modification.
2. Support multiple mainstream algorithms: such as PointPillars, SECOND, PV-RCNN, CenterPoint, etc.
3. Implemented in PyTorch for easy integration and deployment.
4. Strong community support and rich training scripts and pre-trained models.

Its main functional modules include data preprocessing (data augmentation, voxelization, etc.), feature extractors (such as PointNet, VoxelNet), Object Detection heads (such as anchor-based or center-based detectors), and post-processing (NMS, conversion to 3D boxes). 

PointPillars is an efficient real-time 3D Object Detection algorithm proposed by NVIDIA in 2019. Its core idea is to convert the sparse point cloud into regular "pillar voxels" (pillars), so that it can be processed using an efficient 2D convolutional neural network.

The name of the paper is: `PointPillars: Fast Encoders for Object Detection from Point Clouds`. The link to the paper is:  https://arxiv.org/abs/1812.05784 . 

The core idea and process are as follows: 

1. Point cloud pillar voxelization: Divide the point cloud into small pillars on the xy plane. Each pillar retains up to N points (truncated if exceeded, zero-padded if insufficient). Represent each pillar as a tensor: [num_pillars, num_points_per_pillar, feature_dim].
2. Pillar Feature Net (Feature Extraction): Use a small MLP (similar to PointNet) for the points in each pillar to extract local features. Aggregate these features into fixed-dimensional features for each pillar.
3. Pseudo Image Generation: Place the pillar features in the corresponding positions to form a "pseudo image". The input size is [C, H, W], which can be directly processed by 2D CNN.
4. 2D Backbone & Detection Head: Use an efficient 2D convolution network (such as FPN) to extract deeper features. Finally, output the 3D bounding box (position, size, orientation, etc.) through the detection head


# Quick Experience 

## Environment Dependencies

After we have an RDK S100, we log in to its environment. 

First, we clone the code repository: 

```
git clone https://github.com/D-Robotics/rdk_model_zoo_s.git
```

We then switched to the following directory: 

```
cd samples\Vision\PointPillars\python
```

Install the relevant dependencies: 

```
pip install -r requirements.txt
```

Download related inputs: 

```
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointPillars/voxels0.npy
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointPillars/voxel_coords0.npy
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointPillars/voxel_num_points0.npy
```

Download model: 

```
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointPillars/AnchorHeadSingle.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointPillars/backbone.hbm
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointPillars/vfe.hbm
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointPillars/PointPillarScatter.onnx
```

## Effect verification

After completing the above steps, we execute the run command on the RDK S100 to verify the set of point cloud inputs provided above 

```
python main.py
```

After the script is executed, a series of outputs will be generated, including `pillar_features.npy`, `spatial_features_2d.npy`, `batch_cls_preds.npy`, `batch_box_preds.npy`. 

We visualize these outputs using the OpenPCDet tool and get the following results: 

![](source/data/1.png)

## Speed Verification

### VFE

We use the following command to verify the speed of the sweet potato heterogeneous model vfe.hbm 

```
hrt_model_exec perf --model_file vfe.hbm --frame_count 100
```

We obtain the following results 

![](source/data/2.png)

* **Frames**: 100
* **Average Latency**: 408.189ms
* **FPS**: 2.449

### Backbone

We use the following command to verify the speed of the sweet potato heterogeneous model backbone.hbm 

```
hrt_model_exec perf --model_file backbone.hbm --frame_count 100
```

We obtain the following results 

![](source/data/3.png)

* **Frames**: 100
* **Average Latency**: 185.581ms
* **FPS**: 5.378