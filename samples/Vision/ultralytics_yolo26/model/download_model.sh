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

# Detect SoC and Board Type
SOC_NAME=$(cat /sys/class/boardinfo/soc_name 2>/dev/null || echo "s100")
BOARD_TYPE=$(cat /sys/class/boardinfo/board_type 2>/dev/null || echo "s100")
SOC_NAME=${SOC_NAME,,}
BOARD_TYPE=${BOARD_TYPE,,}

# Default march to nash-e (S100)
MARCH="nash-e"
if [[ "$BOARD_TYPE" == *"p"* ]]; then
    MARCH="nash-m"
fi

# Allow manual override via environment variable
if [ -n "$RDK_MARCH" ]; then
    MARCH=$RDK_MARCH
fi

echo "Detected SoC: ${SOC_NAME}, Board: ${BOARD_TYPE}"
echo "Selected architecture (MARCH): ${MARCH}"

# Define Base URL
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/YOLO26/${MARCH}"

# Suffix for filename based on MARCH
SUFFIX="nashe"
if [ "$MARCH" == "nash-m" ]; then
    SUFFIX="nashm"
fi

# Define Models (Only Nano size)
MODELS=(
    "yolo26n_detect_${SUFFIX}_640x640_nv12.hbm"
    "yolo26n_seg_${SUFFIX}_640x640_nv12.hbm"
    "yolo26n_pose_${SUFFIX}_640x640_nv12.hbm"
    "yolo26n_cls_${SUFFIX}_224x224_nv12.hbm"
    "yolo26n_obb_${SUFFIX}_640x640_nv12.hbm"
)

# Create directory if it doesn't exist
mkdir -p "${MARCH}"
cd "${MARCH}"

# Download loop
for model in "${MODELS[@]}"; do
    echo "Downloading ${model} from ${BASE_URL}..."
    wget -c "${BASE_URL}/${model}" -O "${model}" || echo "Failed to download ${model}"
done

echo "Download complete. Models are stored in model/${MARCH}/"
