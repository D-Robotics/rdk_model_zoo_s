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

# 1. Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    pip3 install numpy opencv-python
fi

# 2. Run Main Detection Task
echo "Running YOLO26 Detection..."
# Default to bus.jpg if available, else standard asset path
TEST_IMG="./bus.jpg"
if [ ! -f "$TEST_IMG" ]; then
    TEST_IMG="/app/res/assets/bus.jpg"
fi

# If local image missing, try to download or warn
if [ ! -f "$TEST_IMG" ]; then
    echo "Warning: Test image not found. Using dummy path."
fi

# Determine SoC for model path (simple heuristic)
SOC_NAME=$(cat /sys/class/boardinfo/soc_name 2>/dev/null || echo "s100")
SOC_NAME=${SOC_NAME,,} # to lowercase

# Construct default model path
# Note: This path should match the one in main.py or be a valid path
MODEL_PATH="../../model/${SOC_NAME}/basic/yolo26n_detect_640x640_nv12.hbm"

echo "Using SoC: ${SOC_NAME}"
echo "Model Path: ${MODEL_PATH}"

# Run script
python3 main.py --task detect --model-path "${MODEL_PATH}" --test-img "${TEST_IMG}"
