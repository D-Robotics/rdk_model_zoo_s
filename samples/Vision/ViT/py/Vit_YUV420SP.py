#!/user/bin/env python

# Copyright (c) 2025, SkyXZ D-Robotics.
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

# 注意: 此程序在RDK板端端运行
# Attention: This program runs on RDK board.

# pip install scipy

import os
import cv2
import numpy as np
# scipy
try:
    from scipy.special import softmax
except:
    print("scipy is  not installed, installing.")
    os.system("pip install scipy")
    from scipy.special import softmax

# hobot_dnn
try:
    from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
except:
    print("Your python environment is not ready, please use system python3 to run this program.")
    exit()

from time import time
import argparse
import logging


# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_ViT")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='../source/reference_hbm_models/vit_cifar10_batch1_int8.hbm', 
                        help="""Path to BPU Quantized *.hbm Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--test-img', type=str, default='../source/imgs/test_img/airplane_0000.png', help='Path to Load Test Image.')
    parser.add_argument('--top-k', type=int, default=5, help='Top K predictions to show.')
    parser.add_argument('--input-size', type=int, default=224, help='Input image size for ViT model.')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size for ViT model.')
    opt = parser.parse_args()
    logger.info(opt)

    # quick demo
    if not os.path.exists(opt.model_path):
        print(f"file {opt.model_path} does not exist. Please check the model path.")
        exit(1)
    model = ViT_Classify(opt)
    img = cv2.imread(opt.test_img)
    if img is None:
        raise ValueError(f"Load image failed: {opt.test_img}")
        exit()
    # 准备输入数据
    input_tensor = model.preprocess(img)
    # 推理
    outputs = model.c2numpy(model.forward(input_tensor))
    # 后处理
    results = model.postProcess(outputs, opt.top_k)
    # 打印结果
    logger.info("\033[1;32m" + "Classification Results: " + "\033[0m")
    for i, (class_id, score) in enumerate(results):
        if class_id < len(cifar10_names):
            print(f"\033[33mTop {i+1}: {cifar10_names[class_id]} -> {score:.4f}\033[0m")
        else:
            print(f"\033[33mTop {i+1}: Unknown class {class_id} -> {score:.4f}\033[0m")

class ViT_Classify():
    def __init__(self, opt):
        # 加载BPU的hbm模型, 打印相关参数
        # Load the quantized *.hbm model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(opt.model_path)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(opt.model_path))
            logger.error("You can download the model file from the following docs: ./source/reference_hbm_models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_output in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_output.name}, type={quantize_output.properties.dtype}, shape={quantize_output.properties.shape}")

        # 获取输入输出参数
        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[1:3]
        self.input_C = self.quantize_model[0].inputs[0].properties.shape[3]
        logger.info(f"{self.input_H = }, {self.input_W = }, {self.input_C = }")
        
        # 检查是否有两个输入（Y和UV）
        if len(self.quantize_model[0].inputs) == 2:
            self.has_yuv_input = True
            logger.info("Model has YUV dual input format")
        else:
            self.has_yuv_input = False
            logger.info("Model has single input format")

        # 获取输出类别数
        self.num_classes = self.quantize_model[0].outputs[0].properties.shape[1]
        logger.info(f"{self.num_classes = }")

        self.input_image_size = opt.input_size
        self.patch_size = opt.patch_size
        logger.info(f"Input size: {self.input_image_size}, Patch size: {self.patch_size}")

        self.output_scale = 1.0
        logger.info("Output is already float32 format, no dequantization needed")

    def preprocess(self, img):
        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_LINEAR)
        
        # 转换为YUV420SP格式
        input_tensor = self.bgr2nv12(input_tensor)
        
        logger.debug("\033[1;31m" + f"pre process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        
        return input_tensor

    def bgr2nv12(self, bgr_img):
        begin_time = time()
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        
        # BGR转YUV420P
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        
        # 分离Y和UV分量
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        
        # 组合为NV12格式
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        
        logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return nv12

    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs

    def c2numpy(self, outputs):
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

    def postProcess(self, outputs, top_k=5):
        begin_time = time()
        
        logits = outputs[0].reshape(-1)
        logits_float32 = logits.astype(np.float32) * self.output_scale
        
        # Softmax计算概率
        probabilities = softmax(logits_float32)
        
        # 获取top-k预测结果
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_results = [(int(idx), float(probabilities[idx])) for idx in top_indices]
        
        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        
        return top_results

# CIFAR-10类别名称
cifar10_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

if __name__ == "__main__":
    main() 