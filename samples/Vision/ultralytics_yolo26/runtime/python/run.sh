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
    TEST_IMG="../../../resource/assets/bus.jpg"
fi

# Determine MARCH for model path
BOARD_TYPE=$(cat /sys/class/boardinfo/board_type 2>/dev/null || echo "s100")
BOARD_TYPE=${BOARD_TYPE,,}

MARCH="nash-e"
SUFFIX="nashe"
if [[ "$BOARD_TYPE" == *"p"* ]]; then
    MARCH="nash-m"
    SUFFIX="nashm"
fi

# Construct default model path
MODEL_PATH="../../model/${MARCH}/yolo26n_detect_${SUFFIX}_640x640_nv12.hbm"

# 3. Download model if missing
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at ${MODEL_PATH}, attempting to download..."
    pushd ../../model/ > /dev/null
    ./download_model.sh
    popd > /dev/null
fi

echo "Using MARCH: ${MARCH}"
echo "Model Path: ${MODEL_PATH}"
echo "Test Image: ${TEST_IMG}"

# Run script
python3 main.py --task detect --model-path "${MODEL_PATH}" --test-img "${TEST_IMG}"
