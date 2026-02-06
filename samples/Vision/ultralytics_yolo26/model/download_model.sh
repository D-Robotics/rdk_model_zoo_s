#!/bin/bash

# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# Detect SoC
SOC_NAME=$(cat /sys/class/boardinfo/soc_name 2>/dev/null || echo "s100")
SOC_NAME=${SOC_NAME,,} # to lowercase

echo "Detected SoC: ${SOC_NAME}"

# Define Base URL based on SoC
# S600 uses s600 path, others (S100/X5) use s100 path as per current logic
if [ "$SOC_NAME" == "s600" ]; then
    BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/YOLO26"
else
    BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/YOLO26"
fi

# Define Models to download
MODELS=(
    "yolo26n_detect_640x640_nv12.hbm"
    "yolo26n_seg_640x640_nv12.hbm"
    "yolo26n_pose_640x640_nv12.hbm"
    "yolo26n_cls_224x224_nv12.hbm"
    "yolo26n_obb_640x640_nv12.hbm"
)

# Download loop
for model in "${MODELS[@]}"; do
    echo "Downloading ${model}..."
    wget -c "${BASE_URL}/${model}" -O "${model}" || echo "Failed to download ${model}"
done

echo "Download complete."
