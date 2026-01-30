English | [简体中文](./README_cn.md)

# YOLO26 Model Conversion and Compilation Guide

This directory provides tools and instructions for converting YOLO26 models (originated from the Ultralytics framework) into BPU-quantized models compatible with D-Robotics RDK hardware.

**Supported Platforms**:
- **RDK S100/S100P (Nash)**: Generates `.hbm` model files.

## Model Compilation Environment

To convert the model, you need to use the **OpenExplore Docker Environment**.

### 1. Install Docker
*   Follow the official instructions to install and verify: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
*   Verification:
    ```bash
    sudo docker --version
    sudo docker run --rm hello-world
    ```

### 2. Obtain and Load the Offline Image
*   **Download Image**: Please visit the [D-Robotics Developer Documentation](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F) to download the CPU version of the Docker image adapted for the RDK S100 series.
*   **Load Image**:
    ```bash
    sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
    ```
    *Note: Replace the filename with the one you actually downloaded.*

### 3. Start the Container
It is recommended to use the following command to start the container, mounting the current working directory to the container and increasing the shared memory to avoid memory issues during compilation.

```bash
# Assuming you are currently in the rdk_model_zoo_s root directory
sudo docker run -it --rm \
 --network host \
 --shm-size=15g \
 -v "$(pwd)":/workspace \
 --workdir /workspace \
 <docker-image-name> /bin/bash
```
*   `<docker-image-name>`: Use `sudo docker images` to view the name and tag of the loaded image.

---

## Conversion Workflow

### 1. One-Click Conversion Script (Recommended)

We provide the `mapper.py` script, which automates the entire process, including calibration data preparation, configuration file generation, and calling the compiler (`hb_compile`). Ensure you are **inside the Docker container**.

**Prerequisites**:
- An ONNX model exported for BPU compatibility (refer to `onnx_export/`).
- A folder containing 20~50 images for quantization calibration (`.jpg` or `.png`).

**Run Conversion**:

Once inside the container, navigate to the directory containing `mapper.py`:
```bash
cd samples/Vision/yolo26/conversion
```

**For RDK S100 (Nash-E)**:
```bash
python3 mapper.py --onnx yolo26n.onnx --cal-images ./cal_images --march nash-e
```

**For RDK S100P (Nash-M)**:
```bash
python3 mapper.py --onnx yolo26n.onnx --cal-images ./cal_images --march nash-m
```

### 2. Script Argument Descriptions

`mapper.py` exposes several common parameters to meet customization needs:

```bash
python3 mapper.py -h
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--onnx` | Path to the original floating-point ONNX model. | Required |
| `--cal-images` | Path to the directory containing calibration images (20~50 recommended). | `./cal_images` |
| `--march` | **Key Parameter**: Specifies the target architecture.<br>`nash-e`: RDK S100<br>`nash-m`: RDK S100P | `nash-e` |
| `--quantized` | Quantization precision: `int8` (recommended) or `int16`. | `int8` |
| `--jobs` | Number of concurrent tasks during model compilation. | 16 |
| `--optimize-level` | Compiler optimization level.<br>Nash: `O0`-`O2` | `O3` |
| `--cal-sample` | Whether to sample images from the directory. | `True` |
| `--save-cache` | Whether to preserve temporary files from the compilation process. | `False` |

---

## Troubleshooting
*   **Permission Issues**: Errors when copying files back to the host machine. Check file ownership or use `sudo chown -R`.
*   **Memory/IPC Errors**: Ensure the `--shm-size=15g` parameter was added when starting the container.

## License
Tools in this directory follow the [Apache 2.0 License](../../../../LICENSE).