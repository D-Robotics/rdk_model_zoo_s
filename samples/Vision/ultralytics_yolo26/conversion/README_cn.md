[English](./README.md) | 简体中文

# YOLO26 模型转换与编译指南

本目录提供了将 YOLO26 模型（源自 Ultralytics 框架）转换为适配地瓜机器人（D-Robotics）RDK 硬件的 BPU 量化模型的工具与说明。

**支持平台**:
- **RDK S100/S100P (Nash)**: 生成 `.hbm` 模型。

## 模型编译环境

为了转换模型，您需要使用 **OpenExplore Docker 环境**。

### 1. 安装 Docker
*   按照官方说明安装并验证： [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
*   验证：
    ```bash
    sudo docker --version
    sudo docker run --rm hello-world
    ```

### 2. 获取并加载离线镜像
*   **下载镜像**: 请访问 [D-Robotics 开发者文档](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F) 下载适配 RDK S100 系列的 CPU 版本 Docker 镜像。
*   **加载镜像**:
    ```bash
    sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
    ```
    *注意：请将文件名替换为您实际下载的文件名。*

### 3. 启动容器
建议使用以下命令启动容器，将当前工作目录挂载到容器中，并增大共享内存以避免编译过程中的内存问题。

```bash
# 假设您当前位于 rdk_model_zoo_s 根目录
sudo docker run -it --rm \
 --network host \
 --shm-size=15g \
 -v "$(pwd)":/workspace \
 --workdir /workspace \
 <docker-image-name> /bin/bash
```
*   `<docker-image-name>`: 使用 `sudo docker images` 查看加载后的镜像名称和标签。

---

## 转换流程

### 1. 一键转换脚本 (推荐)

我们提供了 `mapper.py` 脚本，可以自动完成校准数据准备、配置文件生成以及调用编译器 (`hb_compile`) 的全过程。请确保您已进入 **Docker 容器**。

**准备工作**:
- 已经导出为 BPU 适配的 ONNX 模型（参考 `onnx_export/`）。
- 准备一个文件夹，包含 20~50 张用于量化校准的图片（`.jpg` 或 `.png`）。

**运行转换**:

进入容器后，切换到 `mapper.py` 所在目录：
```bash
cd samples/Vision/yolo26/conversion
```

**针对 RDK S100 (Nash-E)**:
```bash
python3 mapper.py --onnx yolo26n.onnx --cal-images ./cal_images --march nash-e
```

**针对 RDK S100P (Nash-M)**:
```bash
python3 mapper.py --onnx yolo26n.onnx --cal-images ./cal_images --march nash-m
```

### 2. 脚本参数说明

`mapper.py` 暴露了一些常用参数以满足定制需求：

```bash
python3 mapper.py -h
```

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--onnx` | 原始浮点 ONNX 模型的路径。 | 必填 |
| `--cal-images` | 包含校准图片的目录路径（建议 20~50 张）。 | `./cal_images` |
| `--march` | **核心参数**: 指定目标架构。<br>`nash-e`: RDK S100<br>`nash-m`: RDK S100P | `nash-e` |
| `--quantized` | 量化精度：`int8` (推荐) 或 `int16`。 | `int8` |
| `--jobs` | 模型编译时的并发任务数。 | 16 |
| `--optimize-level` | 编译器优化等级。<br>Nash: `O0`-`O2` | `O3` |
| `--cal-sample` | 是否从目录中采样图片。 | `True` |
| `--save-cache` | 是否保留编译过程中的临时文件。 | `False` |

---

## 常见问题
*   **权限问题**：宿主机复制回文件时出现权限错误，检查文件属主或使用 `sudo chown -R`。
*   **内存/IPC 报错**：请确保启动容器时添加了 `--shm-size=15g` 参数。

## License
本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。
