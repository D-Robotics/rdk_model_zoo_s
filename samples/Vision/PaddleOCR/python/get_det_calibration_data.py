#===----------------------------------------------------------------------===#
#
# Copyright (C) 2025 D-Robotics. All rights reserved.
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
#
# Contact: shukun.huang@d-robotics.cc
#
#===----------------------------------------------------------------------===#

import numpy as np
import os
import cv2
import concurrent.futures
import psutil

# 模型输入尺寸
model_input_height = 640
model_input_width = 640

# 支持的图片扩展名列表
regular_process_list = [
    ".rgb",
    ".rgbp",
    ".bgr",
    ".bgrp",
    ".yuv",
    ".feature",
    ".cali",
]

def calibration_transformers():
    """
    定义图片预处理步骤：
        1. 调整图片大小并填充到 640x640
        2. 转换通道顺序 (HWC -> CHW)
        3. 归一化 (减去均值并除以标准差)
    """
    def pad_resize(image, target_size):
        h, w, c = image.shape
        scale = min(target_size[0] / h, target_size[1] / w)
        resized_h, resized_w = int(h * scale), int(w * scale)
        resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded_image = np.zeros((target_size[0], target_size[1], c), dtype=np.uint8)
        padded_image[:resized_h, :resized_w, :] = resized_image
        return padded_image

    def hwc_to_chw(image):
        return np.transpose(image, (2, 0, 1))

    def normalize(image, mean, std):
        image = image.astype(np.float32) / 255.0
        mean = mean.reshape((3, 1, 1)) 
        std = std.reshape((3, 1, 1))    
        image -= mean
        image /= std
        return image

    return [
        lambda img: pad_resize(img, (model_input_height, model_input_width)),
        hwc_to_chw,
        lambda img: normalize(img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
    ]

transformers = calibration_transformers()

def apply_transformers(image, transformers):
    """
    依次应用预处理步骤
    """
    for transform in transformers:
        image = transform(image)
    return image

def regular_preprocess(src_file, transformers, dst_dir, pic_ext, saved_data_type):
    """
    常规图片预处理
    """
    print(f"Processing {src_file}")
    image = cv2.imread(src_file)
    if image is None:
        print(f"Warning: Unable to read image {src_file}")
        return
    image = apply_transformers(image, transformers)
    filename = os.path.basename(src_file)
    short_name, _ext = os.path.splitext(filename)
    pic_name = os.path.join(dst_dir, short_name + pic_ext + '.npy')
    print("write:", pic_name)
    dtype = np.float32 if saved_data_type == 'float32' else np.uint8
    np.save(pic_name, image.astype(dtype))

def main():
    """
    主函数：处理图片并保存为 .npy 文件
    """
    src_dir = "dataset"  # 源图片目录
    dst_dir = "det_calibration_data"  # 目标目录
    pic_ext = ".cali"  # 图片扩展名
    cal_img_num = 100  # 需要处理的图片数量
    saved_data_type = "float32"  # 保存数据的类型

    pic_num = 0
    os.makedirs(dst_dir, exist_ok=True)

    if pic_ext.strip().split('_')[0] in regular_process_list:
        print("Regular preprocess")
        parallel_num = min(psutil.cpu_count(), 10)
        print(f"Init {parallel_num} processes")
        for src_name in sorted(os.listdir(src_dir)):
            pic_num += 1
            if pic_num > cal_img_num:
                break
            src_file = os.path.join(src_dir, src_name)
            regular_preprocess(src_file, transformers, dst_dir, pic_ext, saved_data_type)
    else:
        raise ValueError(f"invalid pic_ext {pic_ext}")

if __name__ == '__main__':
    main()