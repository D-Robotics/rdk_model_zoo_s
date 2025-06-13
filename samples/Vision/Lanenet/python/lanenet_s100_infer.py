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

import numpy as np
import cv2
from hobot_dnn import pyeasy_dnn

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    return img
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

def main():
    # 路径根据实际情况修改
    model_path = "lanenet256x512.hbm"
    img_path = "input.jpg"
    output_dir = "output"
    import os
    os.makedirs(output_dir, exist_ok=True)

       # 1. 加载图片
    origin_img = load_image(img_path)
    resize_height, resize_width = 256, 512

    # 2. 手动前处理
    img_input = preprocess_image(origin_img, resize_height, resize_width)
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)  # (1, C, H, W), float32


    # 2. 加载模型
    model =  pyeasy_dnn.load(model_path)[0]

    # 3. 推理
    outputs = model.forward(img_input)

    # 4. 处理输出
    # 假设输出 shape = (1, H, W) 或 (H, W)
    instance_pred= np.array(outputs[0].buffer, dtype=np.float32).reshape((3,256, 512)).squeeze()   # 根据实际输出形状调整 # 根据实际输出形状调整
    binary_pred = np.array(outputs[1].buffer, dtype=np.float32).reshape((256, 512)) 
    
    # 6. 保存结果
    instance_output_path = os.path.join(output_dir, "instance_pred.png")
    binary_output_path = os.path.join(output_dir, "binary_pred.png")
    cv2.imwrite(instance_output_path, (instance_pred * 255).transpose(1, 2, 0).astype(np.uint8))
    cv2.imwrite(binary_output_path, (binary_pred * 255).astype(np.uint8))
    print("Results saved to:", instance_output_path, "and", binary_output_path)


if __name__ == "__main__":
    main()