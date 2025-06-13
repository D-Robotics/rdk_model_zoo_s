"""
 Copyright (c) 2021-2024 D-Robotics Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
# 配置参数
dataset_dir = "dataset"  # 你的图片文件夹
output_dir = "cal_data"  # 输出npy文件夹
resize_height = 256  # 根据你的需求修改
resize_width = 512   # 根据你的需求修改
def preprocess_image(image, resize_height, resize_width):
    # 1. Resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)  # (W, H)
    # 2. ToTensor: HWC -> CHW, [0,255] -> [0,1]
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (C, H, W)
    # 3. Normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = (image - mean) / std
    return image
def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    return img
data_transform = transforms.Compose([
    transforms.Resize((resize_height, resize_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for fname in os.listdir(dataset_dir):
    if fname.lower().endswith('.jpg'):
        img_path = os.path.join(dataset_dir, fname)
        origin_img= load_image(img_path)
        resize_height, resize_width = 256, 512
        img_input = preprocess_image(origin_img, resize_height, resize_width)
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
        # img = Image.open(img_path).convert('RGB')
        print(img_input)
        npy_name = os.path.splitext(fname)[0] + ".npy"
        np.save(os.path.join(output_dir, npy_name), img_input)
        print(f"Saved {npy_name}")

print("All images processed and saved as .npy.")