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
import cv2
import pyclipper
from hobot_dnn import pyeasy_dnn
import matplotlib.pyplot as plt
import os
import glob
# 加载图像函数
def load_image(img_path):
    """加载图像"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    return img

# 确保图像宽高为偶数
def resize_to_640(image):
    """调整图像大小为偶数"""
    resized_image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    return resized_image

# 转换图像为 NV12 格式
def bgr2nv12_opencv(image):
    """将 BGR 图像转换为 NV12 格式"""
    height, width = image.shape[:2]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
    nv12 = np.zeros_like(yuv420p)
    nv12[: height * width] = y
    nv12[height * width :] = uv_packed
    return nv12

# 膨胀轮廓
def dilate_contours(contours, ratio_prime=2.7):
    """膨胀轮廓"""
    dilated_polys = []
    for poly in contours:
        poly = poly[:, 0, :]  # 提取多边形的顶点坐标
        arc_length = cv2.arcLength(poly, True)  # 计算多边形的周长
        if arc_length == 0:
            continue
        D_prime = (cv2.contourArea(poly) * ratio_prime / arc_length)  # 计算膨胀量
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        dilated_poly = np.array(pco.Execute(D_prime))
        if dilated_poly.size == 0 or dilated_poly.dtype != np.int_ or len(dilated_poly) != 1:
            continue
        dilated_polys.append(dilated_poly)
    return dilated_polys

# 获取边界框
def get_bounding_boxes(dilated_polys, min_area=100):
    """从膨胀的多边形中获取边界框"""
    boxes_list = []
    for cnt in dilated_polys:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = cv2.minAreaRect(cnt)  # 计算最小外接矩形
        box = cv2.boxPoints(rect).astype(np.int_)
        boxes_list.append(box)
    return boxes_list

# 裁剪并旋转图像
def crop_and_rotate_image(img, box):
    """根据边界框裁剪并旋转图像"""
    rect = cv2.minAreaRect(box)
    box = cv2.boxPoints(rect).astype(np.intp)
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = rect[2]
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    if angle >= 45:
        rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated = warped
    return rotated
def resize_norm_image(img, input_size):
    """Resize and normalize the image."""
    image_resized = cv2.resize(img, dsize=(input_size[1], input_size[0]))
    image_resized = (image_resized / 255.0).astype(np.float32)
    input_image = np.zeros((image_resized.shape[0], image_resized.shape[1], 3), dtype=np.float32)
    input_image[:image_resized.shape[0], :image_resized.shape[1], :] = image_resized
    input_image = image_resized[:, :, [2, 1, 0]]  # bgr->rgb
    input_image = input_image[None].transpose(0, 3, 1, 2)  # NHWC -> HCHW
    return input_image
# 保存裁剪后的图像为 .npy 文件# 修改保存函数以支持文件名
def save_cropped_images_as_npy(cropped_images, input_size, output_dir, img_path):
    """Resize cropped images and save them as .npy files."""
    base_name = os.path.splitext(os.path.basename(img_path))[0]  # 获取图片的基本名称
    for i, cropped_img in enumerate(cropped_images):
        resized_img = resize_norm_image(cropped_img, input_size)
        output_path = os.path.join(output_dir, f"{base_name}_cropped_{i}.npy")
        np.save(output_path, resized_img)
        print(f"Saved: {output_path}")

        
# 主推理流程
def run_detection_pipeline_for_folder(model_path, img_folder,output_dir="cropped_images_npy"):
    """运行检测模型并处理文件夹中的所有图片"""
    # 加载模型
    detection_model = pyeasy_dnn.load(model_path)
    input_size = (48, 320)
    

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历文件夹中的所有图片
    img_paths = glob.glob(os.path.join(img_folder, "*.jpg"))  # 可根据需要修改图片格式
    for img_path in img_paths:
        print(f"Processing: {img_path}")
        try:
            # 加载图像
            origin_img = load_image(img_path)

            # 预处理图像
            reshape_img = resize_to_640(origin_img)
            nv12_image = bgr2nv12_opencv(reshape_img)

            # 推理
            outputs = detection_model[0].forward(nv12_image)
            preds = np.array(outputs[0].buffer, dtype=np.float32).reshape(1, *reshape_img.shape[:2])
            preds = np.where(preds > 0.5, 255, 0).astype(np.uint8).squeeze()

            # 后处理
            contours, _ = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dilated_polys = dilate_contours(contours)
            boxes_list = get_bounding_boxes(dilated_polys)

            # 裁剪图像
            cropped_images = []
            for box in boxes_list:
                cropped_img = crop_and_rotate_image(origin_img, box)
                cropped_images.append(cropped_img)

            # 保存裁剪后的图像
            if not cropped_images:
                print(f"No valid boxes found in {img_path}.")
                continue
            
            save_cropped_images_as_npy(cropped_images, input_size, output_dir, img_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# 示例调用
model_path = 'cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm'
img_path = "cal_data" # 这里是包含图片的文件夹路径
output_dir = "cropped_images_npy" # 输出目录
cropped_images = run_detection_pipeline_for_folder(model_path, img_path,output_dir)
