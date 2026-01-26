# YOLO26 模型说明

本目录描述 YOLO26 在本 Model Zoo 中的完整使用流程，包括：算法介绍、模型转换、运行时推理（Python）、可复用的前后处理接口说明，以及模型评估步骤。

---

## 算法介绍（Algorithm Overview）



### 算法功能
YOLO26 能完成以下任务：

- **Detection**: 通用目标检测 (COCO 数据集)
- **Segmentation**: 实例分割
- **Pose**: 人体关键点检测
- **Classification**: 图像分类
- **OBB**: 旋转目标检测
---

## 目录结构（Directory Structure）

本目录包含：

```bash
.
|-- conversion                          # 模型转换流程
|   `-- README.md                       # 模型转换使用说明
|-- evaluator                           # 模型评估相关内容
|   `-- README.md                       # 模型评估说明
|-- model                               # 模型文件及下载信息
|   `-- README.md                       # 模型说明
|-- runtime                             # 模型推理示例
|   `-- python                          # Python 推理示例
|       |-- README.md                   # Python 推理示例使用说明
|       |-- README_cn.md                # Python 推理示例使用说明 (中文)
|       |-- main.py                     # 统一推理入口脚本
|       |-- main_detect.py              # 检测任务入口脚本
|       |-- main_seg.py                 # 分割任务入口脚本
|       |-- main_pose.py                # 姿态任务入口脚本
|       |-- main_cls.py                 # 分类任务入口脚本
|       |-- main_obb.py                 # OBB任务入口脚本
|       |-- yolo26_*.py                 # 各任务的封装类
|       `-- ...
`-- README_cn.md                        # YOLO26 示例整体说明 (本文件)
```

---

## 快速体验（QuickStart）

为了便于用户快速上手体验，提供了 Python 脚本可直接运行。

### Python

- 进入`runtime`目录下的`python`目录，运行特定任务的脚本：
    ```bash
    cd runtime/python/
    
    # 安装依赖
    pip install -r requirements.txt
    
    # 运行目标检测示例
    python main_detect.py
    ```
- 若想了解`python`代码的详细使用方法，或 step by step 运行模型请参考`runtime/python/README_cn.md`。

---

## 模型转换（Model Conversion）

- ModelZoo 已提供适配完成的 HBM 模型文件，运行时脚本会在缺失时尝试自动下载。如不关心模型转换流程，**可跳过本小节**。 Coming soon...

- 如需自定义模型转换参数，或了解完整的模型转换流程，请参考`conversion/README.md`。

---

## 模型推理（Runtime）

YOLO26 模型推理示例提供 Python 实现方式，支持快速验证模型效果与算法流程。

### Python 版本

    - 以脚本形式提供，适合快速验证模型效果与算法流程;
    - 示例中展示了模型加载、推理执行、后处理以及结果可视化的完整过程;
    - 支持 YOLO26 的全系任务 (Detect/Seg/Pose/Cls/OBB);
    - 具体使用方法、参数说明及接口说明请参考 `runtime/python/README_cn.md`;

---

## 模型评估（Evaluator）

`evaluator/` 用于模型精度、性能及数值一致性评估，详细说明请参考该目录。

---

## License
遵循 Model Zoo 顶层 License。
