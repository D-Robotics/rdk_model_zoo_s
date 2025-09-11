# MobileNetV2 图像分类示例

本示例展示如何使用基于 BPU 部署的 `MobileNetV2` 模型进行图像分类任务，使用 `HB_HBMRuntime` 进行推理。

## 环境依赖
本样例无特殊环境需求，只需确保安装了pydev中的环境依赖即可。
```bash
pip install -r ../../requirements.txt
```

## 目录结构

```text
.
├── mobilenetv2.py              # 主推理脚本
└── README.md                   # 使用说明
```

## 参数说明
| 参数           | 说明                                                     | 默认值                                      |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | 模型文件路径（.hbm 格式）                                  | `/opt/hobot/model/s100/basic/mobilenetv2_224x224_nv12.hbm`                 |
| `--test-img`   | 测试图片路径                                              | `/app/res/assets/zebra_cls.jpg`                |
| `--label-file` | 类别标签映射文件路径（dict 格式）                           | `/app/res/labels/imagenet1000_clsidx_to_labels.txt` |
| `--priority`   | 模型优先级（0~255，越大优先级越高）                         | `0`                                         |
| `--bpu-cores`  | 推理使用的 BPU 核心编号列表（如 `--bpu-cores 0 1`）         | `[0]`                                       |


## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        python mobilenetv2.py
        ```
    - 指定参数运行
        ```bash
        python mobilenetv2.py \
        --model-path mobilenetv2_224x224_nv12.hbm \
        --test-img /app/res/assets/zebra_cls.jpg \
        --label-file /app/res/labels/imagenet1000_clsidx_to_labels.txt
        ```
- 查看结果
    ```bash
    Top-5 Predictions:
    zebra: 0.8916
    tiger, Panthera tigris: 0.0028
    hartebeest: 0.0018
    jaguar, panther, Panthera onca, Felis onca: 0.0016
    tiger cat: 0.0016
    ```

## 注意事项
- 若指定模型路径不存在，程序将尝试自动下载模型。

- 输出结果显示 top-K 概率最高的类别。

- 如需了解更多部署方式或模型支持情况，请参考官方文档或联系平台技术支持。
