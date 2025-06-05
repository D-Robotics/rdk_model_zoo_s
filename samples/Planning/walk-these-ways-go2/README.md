English| [简体中文](./README_cn.md)

# Preface
This year, humanoid robots and quadruped robots have been frequently in the spotlight, becoming hot topics. For these types of robots, motion control is undoubtedly one of the core technologies. Reinforcement learning, being a prominent method for motion control, happens to fully leverage the inference capabilities of the RDK series boards. Therefore, we attempted to deploy a reinforcement learning model on the RDK S100 and explore its application on real robot platforms.

The robot used in this experiment is the Go2 quadruped robot from Unitree Robotics. The inference device is the RDK S100 (running a compact MLP model with minimal resource usage, see resource usage at the end). We chose to reproduce the `walk-these-ways-go2` project, which is based on the "Walk These Ways" paper published at the Conference on Robot Learning in 2022. The authors proposed a novel method to learn a single policy that encodes a structured set of motor behaviors, addressing training tasks from different perspectives. This approach, known as Multiplicity of Behavior (MoB), enhances generalization and enables the robot to select the most suitable strategy in real-time tasks or new environments without retraining. The project offers a complete engineering implementation, making it suitable for reproduction and optimization. The overall process consists of three steps: Model Training → Model Quantization → On-board Deployment.

Paper: https://arxiv.org/abs/2212.03238

# 1. Model Training

## 1.1 Training Resource Reference
![](https://developer.d-robotics.cc/api/v1/static/imgData/1744198200300.jpg)

## 1.2 Environment Setup
Before setting up the environment, please ensure the following:

1. The training device is equipped with a GPU (recommended ≥ 8GB VRAM if you want to visualize the training process).
2. GPU driver is properly installed and working.
3. CUDA is installed and functioning correctly.
4. It's recommended to use conda to avoid environment conflicts.
5. For installing GPU drivers, CUDA, and conda, please refer to online tutorials.

```bash
# Create and activate environment (Python 3.8 is used; 3.10 not tested)
conda create -n IsaacGym python=3.8
conda activate IsaacGym

# Clone repo and install dependencies
git clone https://github.com/wunuo1/RDK-walk-these-ways-go2.git -b s100
cd RDK-walk-these-ways-go2
pip install -e .

# Download IsaacGym and extract to training environment: https://developer.nvidia.com/isaac-gym
# IsaacGym installation guide: https://junxnone.github.io/isaacgymdocs/install.html
tar zxvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
pip show isaacgym

# Install torch and CUDA (match CUDA version with `nvcc -V`)
conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ml-logger ml-dash --upgrade --no-cache
````

## 1.3 Training

* Set `headless=False` in `train.py` to disable visualization.
* Set `headless=True` if you want to visualize the training (requires GUI support). Note that visualization may slow down training.

Training videos are saved to `RDK-walk-these-ways-go2/runs/gait-conditioned-agility/<date>/train/<id>/videos`.

```bash
# num_envs: number of robots in the training environment (reduce if VRAM is insufficient)
# num_learning_iterations: number of training iterations
python scripts/train.py

# After training, set the model and config path in play.py to preview results
python scripts/play.py
```

## 1.4 Data Visualization

![](https://developer.d-robotics.cc/api/v1/static/imgData/1744198269339.png)

```bash
# Visualize training rewards
# In new terminal, start ml_dash frontend (default port: 3001)
python -m ml_dash.app

# In another terminal, start ml_dash backend in project root (default port: 8081)
python -m ml_dash.server .

# Open browser: localhost:3001
# Username: runs | API: http://localhost:8081 | Access Token: leave blank
# Click profile → select training run → view reward charts in highlighted area
```

## 1.5 Export ONNX Model

During training, checkpoints (`ac_weights`) are saved. Extract the best `adaptation_module` and `body` from the checkpoint with the highest `rew_total` (around 24,000 iterations).

```bash
# Run tool.py to convert selected checkpoint to ONNX
python3 tool.py path_of_ac_weights_xxxx.pt
```

Model architecture and data processing:
![](https://developer.d-robotics.cc/api/v1/static/imgData/1744198308957.jpg)

## 1.6 Reward Function Explanation

![](https://developer.d-robotics.cc/api/v1/static/imgData/1744198343336.jpg)

Reward functions are defined in `corl_rewards.py`. Add new ones using `_reward_xxx` format. Adjust weights in `reward_scales` in `legged_robot_config.py`. Set weight to 0 to disable.

# 2. Model Quantization

## 2.1 Environment Setup

```bash
# Pull S100 toolchain Docker image (request from engineer)

# Extract provided quantization folder (includes config and calibration data)

# Move exported ONNX models to quant folder, mount folder into Docker container
sudo docker run -it --entrypoint="/bin/bash" -v <host_folder_path>:/go2_model_convert_s100 <image_id>
```

## 2.2 Quantization Compilation (Int16)

```bash
cd /go2_model_convert_s100

# Quantize adaptation_module → output: model_output_ad
hb_compile -c ad_config_bpu.yaml --march nash-e

# Quantize body → output: model_output_body
hb_compile -c body_config_bpu.yaml --march nash-e
```

![](https://developer.d-robotics.cc/api/v1/static/imgData/1744198588504.jpg)

# 3. On-Board Deployment

## 3.1 Hardware Preparation

### 3.1.1 Materials List

STL files are in the `bracket` folder of the repo:
![](https://developer.d-robotics.cc/api/v1/static/imgData/1744198642808.png)

### 3.1.2 Device Connection

![](https://developer.d-robotics.cc/api/v1/static/imgData/1744198758699.jpg)

## 3.2 Environment Setup

### 3.2.1 Clone Code and Install Dependencies

```bash
git clone https://github.com/wunuo1/RDK-walk-these-ways-go2 -b s100
cd RDK-walk-these-ways-go2
pip install -e .
pip install torch==1.10.2 torchvision==0.11.3
```

### 3.2.2 Build LCM

```bash
git clone https://github.com/lcm-proj/lcm.git 
cd lcm 
mkdir build 
cd build 
cmake .. 
make 
sudo make install
```

### 3.2.3 Build Unitree SDK

```bash
cd RDK-walk-these-ways-go2/go2_gym_deploy/unitree_sdk2_bin/library/unitree_sdk2 
rm -r build 
sudo ./install.sh 
mkdir build 
cd build 
cmake .. 
make
```

### 3.2.4 Build `lcm_position_go2`

```bash
cd go2_gym_deploy 
rm -r build 
mkdir build 
cd build 
cmake .. 
make -j4
```

### 3.2.5 Connect to Go2 via Ethernet

```bash
# Configure static IP
vim /etc/netplan/01-hobot-net.yaml
eth0:
  dhcp4: no
  dhcp6: no
  addresses:
      - 192.168.123.100/24
  gateway4: 192.168.123.1

# Check Go2 IP (should be 192.168.123.161)
ping 192.168.123.161  

# Check network interface name (e.g., eth0)
ifconfig
```

### 3.2.6 Build Inference Library

```bash
git clone https://github.com/wunuo1/model_task.git -b s100 
mkdir build 
cd build 
cmake .. 
make -j4
cp libmodel_task.so RDK-walk-these-ways-go2/go2_gym_deploy/scripts
```

## 3.3 Run

### 3.3.1 Launch Instructions

```bash
# Launch LCM communication node
cd go2_gym_deploy/build 
sudo ./lcm_position_go2 eth0  

# In another terminal, run control policy
# First press “R2” for standing, press “R2” again to start walking
# Alternatively use: deploy_policy_s100_onnx.py
python3 deploy_policy_s100.py
```

### 3.3.2 Resource Usage

#### 3.3.2.1 Using .bin model

![](https://developer.d-robotics.cc/api/v1/static/imgData/1744198918662.jpg)

* CPU: 28.5% (single core)
* Memory: 3.5%
* BPU: 1%

#### 3.3.2.2 Using ONNX model

![](https://developer.d-robotics.cc/api/v1/static/imgData/1744199406559.jpg)

* CPU: 255.1% (single core)
* Memory: 4.2%

The int16 model reduces CPU usage significantly; BPU usage increases to 1%. Both models perform similarly in motion quality.

### 3.3.3 Demonstration

Video: [https://www.bilibili.com/video/BV154d2YTEn1/](https://www.bilibili.com/video/BV154d2YTEn1/)

# 4. Acknowledgements

Special thanks to the Horizon Robotics Lab for their strong support! Gratitude also to Wang Yucheng and Wang Kaihui for their invaluable assistance!

