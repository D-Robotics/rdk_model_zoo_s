English| [简体中文](./README_cn.md)

# 1. Background Introduction

PointNet is a deep learning model for processing point cloud data, whose status is comparable to that of the CNN network in 2D image processing, and many subsequent deep networks for point cloud data processing bear the influence of PointNet.



## 1.1 Introduction to Point Cloud Data

Since PointNet deals with point cloud data, we first need to know what point cloud data looks like. In fact, so-called point cloud data is a collection composed of a series of points,

Each point represents a three-dimensional coordinate (x, y, z), (Note: Some point cloud data may have six values, including nx, ny, nz in addition to the three coordinates, which form the normal vector.)

A point cloud data composed of N points is actually an Nx3 array.&#x20;

An important characteristic of point cloud data is its disorderliness. Since it is just a collection of points, the order does not matter.

Network design should take this characteristic into account. So what functions have such properties? For example, max, sum, etc., these are all order-independent.&#x20;

PointNet uses the max method because it is simple.



## 1.2 Introduction to PointNet

![](images/image-1.png)

The network structure is very simple, directly inputting point cloud data, then going through several MLP layers for dimensionality increase (in the paper, it is increased to 1024 dimensions), then performing a max operation (taking the max for all points in each dimension), and finally obtaining a 1x1024 dimensional global feature.&#x20;

Based on this feature, classification tasks can be directly performed. In fact, the idea of using several MLPs for dimension elevation and feature extraction is also easy to understand. Suppose there is no dimension elevation operation, after max, there are only 3-dimensional features, which obviously contain too little information. Using this feature for classification or other tasks is clearly not very reliable.

![](images/image.png)

## 1.3 PointNet ONNX

When we export PointNet to ONNX, we can observe that the operators in its ONNX graph include the following:&#x20;

![](images/char_static.png)

As can be seen from the figure above, the main operators include conventional operators such as Conv, BatchNorm, and RELU. These conventional operators are all supported on the RDK S100. Subsequently, we deployed this model on the RDK S100.&#x20;

# 2. RDK S100 Deployment&#x20;

## 2.1 Performance of the model upper board

We converted the model to the HBM format, which is a Digua heterogeneous model and supports the use of Digua BPU computing power.&#x20;

Use the hrt\_model\_exec tool of the RDK S100 version 4.0.3beta to perform model performance testing.

The following result table is obtained:&#x20;

| Number of Threads | Total frame number | Total Delay | Average Delay  | FPS    |
| ----------------- | ------------------ | ----------- | -------------- | ------ |
| 1                 | 100                | 143.63      | 1.43           | 689.61 |
| 2                 | 100                | 216.20      | 2.16           | 914.32 |
| 4                 | 100                | 429.76      | 4.30           | 910.70 |
| 8                 | 100                | 839.86      | 8.35           | 910.84 |

## 2.2 Model Upper Plate Precision

We will evaluate this accuracy as quantization accuracy, i.e., whether the accuracy of the fp32 model after int quantization aligns with that of the original fp32 CPU model.&#x20;

For this quantization, we choose int16 type quantization, and the obtained quantization accuracy is as follows:&#x20;

![](images/PixPin_2025-07-07_20-44-37.jpg)

It can be seen that for trans, the result accuracy is >0.9999 (4 nines), while the pred accuracy is >0.98

## 2.3 Explanation of Model Upper Plate Operation

Code repository reference:https://gitee.com/chenguanzhong/rdk\_-s100\_-point-net\_-official

Let's first install the relevant dependencies:

```
pip install -r requirements.txt
```

Next, we install the Python inference library for BPU:

```
pip install hbmruntime-0.1.0-cp310-cp310-linux_aarch64.whl
```

Rename the model file:

```
mv pointnet.hbm1 pointnet.hbm
```

Ensure that the current folder contains three files: chair.pts, main.py, and pointnet.hbm, which are the data, the main program, and the model file for the BPU board, respectively.&#x20;

The running method is:

```
python main.py
```

After the function is executed, two new image files, chair.png and chair\_res.png, can be seen in the current directory, representing the original point cloud data and the result of point cloud segmentation, respectively.

We see that the resulting point cloud is:

![](chair_res.png)