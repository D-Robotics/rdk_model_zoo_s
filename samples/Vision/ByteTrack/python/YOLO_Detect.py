#!/user/bin/env python

# Copyright (c) 2025, WuChao D-Robotics.
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
    print("scipy is not installed, installing.")
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
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='source/reference_hbm_models/yolo12x_detect_nashe_640x640_nv12.hbm',
                        help="""BPU 量化 *.hbm 模型路径。
                                RDK X3(模块): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(模块): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument('--input', type=str, default='../../../resource/datasets/COCO2017/assets/bus.jpg',
                        help='要加载的测试图像或视频路径。使用 "camera" 表示摄像头。')
    parser.add_argument('--output', type=str, default='py_result.avi',
                        help='处理结果的保存路径，可以是图像或视频文件。')
    parser.add_argument('--classes-num', type=int, default=80, help='检测的类别数量。')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU 阈值。')
    parser.add_argument('--score-thres', type=float, default=0.25, help='置信度阈值。')
    parser.add_argument('--reg', type=int, default=16, help='DFL 回归层。')
    opt = parser.parse_args()
    logger.info(opt)

    # 快速演示：检查模型文件是否存在，如果不存在则下载
    if not os.path.exists(opt.model_path):
        print(f"文件 {opt.model_path} 不存在。正在下载 yolo12n 模型。")
        os.system("wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolo12n_detect_nashe_640x640_nv12.hbm")
        opt.model_path = 'yolo12n_detect_nashe_640x640_nv12.hbm'

    # 实例化模型
    model = YOLO11_Detect(opt)

    # 判断输入是图像、视频还是摄像头
    if opt.input.lower() == 'camera':
        cap = cv2.VideoCapture(0)  # 0 代表默认摄像头
        logger.info("正在从摄像头读取视频流...")
    elif os.path.isfile(opt.input):
        # 检查文件是否是图像文件（通过扩展名判断，这里简化处理）
        if opt.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img = cv2.imread(opt.input)
            if img is None:
                raise ValueError(f"加载图像失败: {opt.input}")

            # 处理单张图像
            input_tensor = model.preprocess_yuv420sp(img)
            outputs = model.c2numpy(model.forward(input_tensor))
            results = model.postProcess(outputs)
            logger.info("\033[1;32m" + "绘制检测结果: " + "\033[0m")
            for class_id, score, x1, y1, x2, y2 in results:
                # 在没有可视化的情况下，可以选择不打印每个检测结果的详细信息，或只打印到日志
                # print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
                draw_detection(img, (x1, y1, x2, y2), score, class_id)
            cv2.imwrite(opt.output, img)
            logger.info("\033[1;32m" + f"结果已保存到: \"./{opt.output}\"" + "\033[0m")
            return # 处理完单张图像后退出
        else:
            # 假定为视频文件
            cap = cv2.VideoCapture(opt.input)
            logger.info(f"正在从视频文件 {opt.input} 读取帧...")
    else:
        raise ValueError(f"无效的输入路径或类型: {opt.input}. 请提供图像文件、视频文件路径或 'camera'。")


    if not cap.isOpened():
        raise IOError(f"无法打开视频源: {opt.input}")

    # 获取视频属性用于输出
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: # 处理某些摄像头或视频流可能报告 FPS 为 0 的情况
        fps = 30 # 默认设置为 30 FPS

    # 定义编解码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Or 'MP4V', 'MJPG', etc.
    out = cv2.VideoWriter(opt.output, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise IOError(f"无法创建视频写入器: {opt.output}. 请检查文件路径和权限。")
    logger.info(f"正在将处理后的视频写入: {opt.output}")

    # 视频帧处理循环
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("视频流结束或无法读取帧。")
            break

        frame_count += 1
        # logger.debug(f"正在处理第 {frame_count} 帧...")

        # 准备输入数据
        input_tensor = model.preprocess_yuv420sp(frame)
        # 推理
        outputs = model.c2numpy(model.forward(input_tensor))
        # 后处理
        results = model.postProcess(outputs)
        # 渲染检测结果
        for class_id, score, x1, y1, x2, y2 in results:
            draw_detection(frame, (x1, y1, x2, y2), score, class_id)

        # 将处理后的帧写入输出视频
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    # 移除关闭所有窗口的代码
    # cv2.destroyAllWindows()
    logger.info("\033[1;32m" + f"处理后的视频已保存到: \"./{opt.output}\"" + "\033[0m")

class YOLO11_Detect():
    def __init__(self, opt):
        # 加载BPU的hbm模型, 打印相关参数
        # Load the quantized *.hbm model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(opt.model_path)
            logger.debug("\033[1;31m" + "加载 D-Robotics 量化模型耗时 = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ 加载模型文件失败: %s"%(opt.model_path))
            logger.error("您可以从以下文档下载模型文件: ./models/download.md")
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> 输入张量" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"输入[{i}], 名称={quantize_input.name}, 类型={quantize_input.properties.dtype}, 形状={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> 输出张量" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"输出[{i}], 名称={quantize_input.name}, 类型={quantize_input.properties.dtype}, 形状={quantize_input.properties.shape}")

        # 将反量化系数准备好, 只需要准备一次
        # prepare the quantize scale, just need to generate once
        self.s_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[3].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[5].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_bboxes_scale.shape=}, {self.m_bboxes_scale.shape=}, {self.l_bboxes_scale.shape=}")

        # DFL求期望的系数, 只需要生成一次
        # DFL calculates the expected coefficients, which only needs to be generated once.
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        # anchors, 只需要生成一次
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80),
                                 np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40),
                                 np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20),
                                 np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)
        logger.info(f"{self.s_anchor.shape = }, {self.m_anchor.shape = }, {self.l_anchor.shape = }")

        # 输入图像大小, 一些阈值, 提前计算好
        self.input_image_size = 640
        self.SCORE_THRESHOLD = opt.score_thres
        self.NMS_THRESHOLD = opt.nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1)
        logger.info("置信度阈值 = %.2f, 非极大值抑制阈值 = %.2f"%(self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f"%self.CONF_THRES_RAW)

        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[1:3]
        logger.info(f"输入高度 = {self.input_H}, 输入宽度 = {self.input_W}")

        self.REG = opt.reg
        print(f"REG = {self.REG}")

        self.CLASSES_NUM = opt.classes_num
        print(f"CLASSES_NUM = {self.CLASSES_NUM}")

    def preprocess_yuv420sp(self, img):
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE
        logger.info(f"预处理类型 = {PREPROCESS_TYPE}")

        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if PREPROCESS_TYPE == RESIZE_TYPE:
            # 利用resize的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST) # 利用resize重新开辟内存节约一次
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.info("\033[1;31m" + f"预处理(resize)耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            # 利用 letter box 的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
            self.y_scale = self.x_scale

            if self.x_scale <= 0 or self.y_scale <= 0:
                raise ValueError("无效的缩放因子。")

            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift

            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift

            input_tensor = cv2.resize(img, (new_w, new_h))
            input_tensor = cv2.copyMakeBorder(input_tensor, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            input_tensor = self.bgr2nv12(input_tensor)
            logger.info("\033[1;31m" + f"预处理(letter box)耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        else:
            logger.error(f"非法的预处理类型 = {PREPROCESS_TYPE}")
            exit(-1)

        logger.debug("\033[1;31m" + f"预处理总耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        logger.info(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}")
        logger.info(f"y_shift = {self.y_shift:.2f}, x_shift = {self.x_shift:.2f}")
        return input_tensor

    def bgr2nv12(self, bgr_img):
        begin_time = time()
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        logger.debug("\033[1;31m" + f"bgr8 到 nv12 转换耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return nv12

    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"推理耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs

    def c2numpy(self, outputs):
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"C 结构转 NumPy 数组耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

    def postProcess(self, outputs):
        begin_time = time()
        # reshape
        s_clses = outputs[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs[1].reshape(-1, self.REG * 4)
        m_clses = outputs[2].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs[3].reshape(-1, self.REG * 4)
        l_clses = outputs[4].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs[5].reshape(-1, self.REG * 4)

        # classify: 利用numpy向量化操作完成阈值筛选(优化版 2.0)
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # 3个Classify分类分支：Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # 3个Bounding Box分支：反量化
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale

        # 3个Bounding Box分支：dist2bbox (ltrb2xyxy)
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32

        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        # xyxy 2 xyhw
        # xy = (dbboxes[:,2:4] + dbboxes[:,0:2])/2.0
        hw = (dbboxes[:,2:4] - dbboxes[:,0:2])
        # xyhw = np.hstack([xy, hw])

        xyhw2 = np.hstack([dbboxes[:,0:2], hw])

        # 分类别nms
        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids==i
            indices = cv2.dnn.NMSBoxes(xyhw2[id_indices,:], scores[id_indices], self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
            for indic in indices.flatten():
                x1, y1, x2, y2 = dbboxes[id_indices,:][indic]
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, self.img_w)
                y2 = min(y2, self.img_h)

                results.append((i, scores[id_indices][indic], x1, y1, x2, y2))

        logger.debug("\033[1;31m" + f"后处理耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        return results

coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img, bbox, score, class_id) -> None:
    """
    在图像上绘制检测到的边界框和标签。

    参数:
        img (np.array): 输入图像。
        bbox (tuple[int, int, int, int]): 包含边界框坐标 (x1, y1, x2, y2) 的元组。
        score (float): 对象的检测分数。
        class_id (int): 检测到对象的类别 ID。
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()