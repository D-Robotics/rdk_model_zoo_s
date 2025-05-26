#!/user/bin/env python

# Copyright (c) 2025, MaChao D-Robotics.
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
    from scipy.optimize import linear_sum_assignment # ByteTrack 使用
except ImportError:
    print("scipy 未安装或不完整，正在安装/升级。")
    os.system("pip install -U scipy") # 确保已安装并更新
    from scipy.special import softmax
    from scipy.optimize import linear_sum_assignment


# hobot_dnn
try:
    from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
except ImportError:
    print("您的 Python 环境未准备好，请使用系统 python3 运行此程序。")
    exit()

from time import time
import argparse
import logging

# 导入 ByteTrack
try:
    from tracker.byte_tracker import BYTETracker
except ImportError:
    print("未找到 ByteTrack。请确保已安装并添加到 PYTHONPATH 中。")
    print("您可以从以下地址克隆: https://github.com/ifzhang/ByteTrack")
    exit()


# 日志模块配置
logging.basicConfig(
    level=logging.INFO, # 修改为 INFO 级别，以减少默认日志的冗余信息
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO_ByteTrack")

# IoU 计算的辅助函数 (如果不直接使用 ByteTrack 中的函数)
def calculate_iou(box1, box2):
    """
    计算两个边界框之间的 IoU。
    边界框格式: [x1, y1, x2, y2]
    """
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='source/reference_hbm_models/yolo12x_detect_nashe_640x640_nv12.hbm',
                        help="""BPU 量化 *.bin 模型路径。
                                 RDK X3(模块): Bernoulli2.
                                 RDK Ultra: Bayes.
                                 RDK X5(模块): Bayes-e.
                                 RDK S100: Nash-e.
                                 RDK S100P: Nash-m.""")
    parser.add_argument('--input', type=str, default='./source/track_test.mp4',
                        help='要加载的测试图像或视频路径。使用 "camera" 表示摄像头。')
    parser.add_argument('--output', type=str, default='outputs/result.mp4',
                        help='处理结果的保存路径，可以是图像或视频文件。')
    parser.add_argument('--classes-num', type=int, default=80, help='检测的类别数量。')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='YOLO NMS 的 IoU 阈值。') 
    parser.add_argument('--score-thres', type=float, default=0.2, help='YOLO 检测的置信度阈值。') 
    parser.add_argument('--reg', type=int, default=16, help='DFL 回归层。')
    
    # ByteTrack 特定参数 
    parser.add_argument('--track-thresh', type=float, default=0.3, help='ByteTrack 跟踪置信度阈值')
    parser.add_argument('--track-buffer', type=int, default=60, help='ByteTrack 丢失轨迹的缓冲帧数')
    parser.add_argument('--match-thresh', type=float, default=0.8, help='ByteTrack 第二次关联的匹配阈值')
    
    opt = parser.parse_args()
    logger.info(opt)

    if not os.path.exists(opt.model_path):
        print(f"文件 {opt.model_path} 不存在。正在下载 yolo12n 模型。")
        os.system("wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolo12n_detect_nashe_640x640_nv12.hbm")
        opt.model_path = 'yolo12n_detect_nashe_640x640_nv12.hbm'

    model = YOLO11_Detect(opt)

    # 初始化 ByteTrack
    # 对于 ByteTrack，frame_rate 很重要。我们将在获取视频源后设置它。
    # 对于单张图像，使用默认值即可。
    class ByteTrackArgs:
        def __init__(self):
            self.track_thresh = opt.track_thresh 
            self.track_buffer = opt.track_buffer
            self.match_thresh = opt.match_thresh
            self.mot20 = False  # MOT20 数据集
            self.frame_rate = 30 # 默认值，对于视频会进行更新

    bytetrack_args = ByteTrackArgs()
    tracker = BYTETracker(bytetrack_args)
    
    cap = None
    is_video_input = False
    person_class_id = 0

    if opt.input.lower() == 'camera':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             raise IOError(f"无法打开摄像头")
        logger.info("正在从摄像头读取视频流...")
        is_video_input = True
        # 使用正确的帧率重新初始化
        bytetrack_args.frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        tracker = BYTETracker(bytetrack_args) 
    elif os.path.isfile(opt.input):
        if opt.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img = cv2.imread(opt.input)
            if img is None:
                raise ValueError(f"加载图像失败: {opt.input}")

            logger.info(f"处理单张图像: {opt.input}")
            input_tensor = model.preprocess_yuv420sp(img) # 在模型中设置 self.img_h, self.img_w 等属性
            outputs = model.c2numpy(model.forward(input_tensor))
            yolo_results = model.postProcess(outputs) # 格式: (class_id, score, x1, y1, x2, y2) 列表

            person_results = [r for r in yolo_results if r[0] == person_class_id]
            logger.info(f"检测到 {len(yolo_results)} 个物体, 其中 person 类别 {len(person_results)} 个。")

            detections_for_bytetrack = []
            if len(person_results) > 0:
                detections_for_bytetrack = np.array(
                    [[r[2], r[3], r[4], r[5], r[1]] for r in person_results] # x1,y1,x2,y2,score
                )
            
            original_img_h, original_img_w = model.img_h, model.img_w
            online_targets = []
            if len(detections_for_bytetrack) > 0:
                # 更新 ByteTrack
                # 由于 detections_for_bytetrack 中的坐标已经是相对于预处理后的图像尺寸，所以传入的 img_info 和 img_size 都应该是原始的图像尺寸，避免在tracker.update 中进行错误的缩放。
                 online_targets = tracker.update(detections_for_bytetrack, 
                                            (frame_height, frame_width), 
                                            (frame_height, frame_width))
            
            logger.info("\033[1;32m" + "绘制跟踪结果: " + "\033[0m")
            for target in online_targets:
                tlwh = target.tlwh
                track_id = target.track_id
                score = target.score 
                # class_id = target.cls
                
                matched_class_id = -1
                best_iou = 0.01 # 设置一个小的阈值以确保有一定的重叠
                
                track_bbox_xyxy = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]

                for yolo_det in person_results: # 使用 person_results 进行匹配
                    yolo_cls, yolo_score, y_x1, y_y1, y_x2, y_y2 = yolo_det
                    yolo_bbox_xyxy = [y_x1, y_y1, y_x2, y_y2]
                    
                    current_iou = calculate_iou(track_bbox_xyxy, yolo_bbox_xyxy)
                    if current_iou > best_iou and abs(score - yolo_score) < 0.1:
                        best_iou = current_iou
                        matched_class_id = yolo_cls
                
                # 仅当我们找到对应的类别时才绘制
                if matched_class_id == person_class_id: 
                    draw_track(img, track_bbox_xyxy, score, matched_class_id, track_id)
                else:
                    logger.debug(f"轨迹 ID {track_id} (预期为 person) 未能匹配或匹配到其他类别。IoU: {best_iou:.2f}")

            cv2.imwrite(opt.output, img)
            logger.info("\033[1;32m" + f"结果已保存到: \"./{opt.output}\"" + "\033[0m")
            return
        else:
            cap = cv2.VideoCapture(opt.input)
            if not cap.isOpened():
                raise IOError(f"无法打开视频文件: {opt.input}")
            logger.info(f"正在从视频文件 {opt.input} 读取帧...")
            is_video_input = True
            # 使用正确的帧率重新初始化
            bytetrack_args.frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            tracker = BYTETracker(bytetrack_args) 
    else:
        raise ValueError(f"无效的输入路径或类型: {opt.input}. 请提供图像文件、视频文件路径或 'camera'。")

    # 单张图片逻辑应该已经处理了这种情况
    if not is_video_input: 
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps 已在 bytetrack_args.frame_rate 中设置

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(opt.output, fourcc, bytetrack_args.frame_rate, (frame_width, frame_height))
    if not out.isOpened():
        raise IOError(f"无法创建视频写入器: {opt.output}. 请检查文件路径和权限。")
    logger.info(f"正在将处理后的视频写入: {opt.output}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("视频流结束或无法读取帧。")
            break

        frame_count += 1
        # logger.debug(f"正在处理第 {frame_count} 帧...") # 可以取消注释以进行调试

        input_tensor = model.preprocess_yuv420sp(frame)
        outputs_raw = model.forward(input_tensor)
        outputs_np = model.c2numpy(outputs_raw)
        yolo_results_all = model.postProcess(outputs_np)

        # --- 只筛选 person 类别 ---
        person_results = [r for r in yolo_results_all if r[0] == person_class_id]
        if frame_count % 30 == 0: # 每30帧打印一次数量信息，避免日志过多
             logger.info(f"帧 {frame_count}: 检测到 {len(yolo_results_all)} 个物体, 其中 person 类别 {len(person_results)} 个。")

        detections_for_bytetrack = []
        if len(person_results) > 0: # 使用筛选后的 person_results
            detections_for_bytetrack = np.array(
                [[r[2], r[3], r[4], r[5], r[1]] for r in person_results] # x1,y1,x2,y2,score
            )
        
        online_targets = []
        if len(detections_for_bytetrack) > 0:
            t1 = time() # 记录更新前的时间
            # 更新 ByteTrack
            # 由于 detections_for_bytetrack 中的坐标已经是相对于预处理后的图像尺寸，所以传入的 img_info 和 img_size 都应该是原始的图像尺寸，避免在tracker.update 中进行错误的缩放。
            online_targets = tracker.update(detections_for_bytetrack, 
                                            (frame_height, frame_width), 
                                            (frame_height, frame_width))
            t2 = time()
            logger.info(f"ByteTrack 更新耗时 = {1000 * (t2 - t1):.2f} ms, 当前在线目标数: {len(online_targets)}")

        for target in online_targets:
            tlwh = target.tlwh
            track_id = target.track_id
            score = target.score

            matched_class_id = -1
            best_iou = 0.01
            track_bbox_xyxy = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]

            for yolo_det in person_results: # 使用 person_results 进行匹配
                yolo_cls, yolo_score, y_x1, y_y1, y_x2, y_y2 = yolo_det
                yolo_bbox_xyxy = [y_x1, y_y1, y_x2, y_y2]
                
                current_iou = calculate_iou(track_bbox_xyxy, yolo_bbox_xyxy)
                if current_iou > best_iou and abs(score - yolo_score) < 0.1:
                    best_iou = current_iou
                    matched_class_id = yolo_cls # 这里应该是 person_class_id (0)
            
            if matched_class_id == person_class_id: # 确保匹配到的确实是 person
                draw_track(frame, track_bbox_xyxy, score, matched_class_id, track_id)
            else:
                # 可选：绘制未匹配的轨迹或记录日志
                # cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), (0,0,255), 2)
                # cv2.putText(frame, f"ID:{track_id} (Unmatched)", (int(tlwh[0]), int(tlwh[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                logger.debug(f"轨迹 ID {track_id} 在第 {frame_count} 帧未能可靠匹配到 YOLO 类别。")


        out.write(frame)
        # 如果您想实时显示:
        # cv2.imshow("RDK YOLO ByteTrack", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    # cv2.destroyAllWindows() # 如果使用了 imshow
    logger.info("\033[1;32m" + f"处理后的视频已保存到: \"./{opt.output}\"" + "\033[0m")

class YOLO11_Detect():
    def __init__(self, opt):
        try:
            begin_time = time()
            self.quantize_model = dnn.load(opt.model_path)
            logger.debug("\033[1;31m" + "加载 D-Robotics 量化模型耗时 = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ 加载模型文件失败: %s"%(opt.model_path))
            logger.error(e) # 打印原始错误信息
            exit(1)

        logger.info("\033[1;32m" + "-> 输入张量" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"输入[{i}], 名称={quantize_input.name}, 类型={quantize_input.properties.dtype}, 形状={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> 输出张量" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs): # 这里应该是 quantize_output
            logger.info(f"输出[{i}], 名称={quantize_input.name}, 类型={quantize_input.properties.dtype}, 形状={quantize_input.properties.shape}")

        # 存储原始图像高宽和模型输入高宽
        self.img_h, self.img_w = 0, 0 
        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[1:3]
        logger.info(f"模型输入高度 = {self.input_H}, 模型输入宽度 = {self.input_W}")
        
        self.s_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[3].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[5].properties.scale_data[np.newaxis, :]
        
        self.weights_static = np.array([i for i in range(opt.reg)]).astype(np.float32)[np.newaxis, np.newaxis, :] # 使用 opt.reg
        
        # 确保使用整数除法 //
        s_feat_h, s_feat_w = self.input_H // 8, self.input_W // 8
        m_feat_h, m_feat_w = self.input_H // 16, self.input_W // 16
        l_feat_h, l_feat_w = self.input_H // 32, self.input_W // 32

        self.s_anchor = np.stack([np.tile(np.linspace(0.5, s_feat_w - 0.5, s_feat_w), reps=s_feat_h),
                                  np.repeat(np.arange(0.5, s_feat_h, 1), s_feat_w)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, m_feat_w - 0.5, m_feat_w), reps=m_feat_h),
                                  np.repeat(np.arange(0.5, m_feat_h, 1), m_feat_w)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, l_feat_w - 0.5, l_feat_w), reps=l_feat_h),
                                  np.repeat(np.arange(0.5, l_feat_h, 1), l_feat_w)], axis=0).transpose(1,0)


        self.SCORE_THRESHOLD = opt.score_thres
        self.NMS_THRESHOLD = opt.nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1) if self.SCORE_THRESHOLD > 0 and self.SCORE_THRESHOLD < 1 else self.SCORE_THRESHOLD 

        self.REG = opt.reg
        self.CLASSES_NUM = opt.classes_num
        
        # 用于将坐标缩放回原始图像
        self.x_scale, self.y_scale = 1.0, 1.0
        self.x_shift, self.y_shift = 0, 0


    def preprocess_yuv420sp(self, img):
        # 确保 self.img_h, self.img_w 设置为原始图像维度
        # 并且 self.x_scale, self.y_scale, self.x_shift, self.y_shift 计算正确
        # 以便 postProcess 将坐标转换回原始图像空间。
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE # 或根据您的模型选择 RESIZE_TYPE
        # logger.info(f"预处理类型 = {PREPROCESS_TYPE}") # 此日志可能比较多余

        #局部计时器
        # begin_time = time() 
        self.img_h, self.img_w = img.shape[0:2] # 存储原始维度
        
        if PREPROCESS_TYPE == RESIZE_TYPE:
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST)
            input_tensor = self.bgr2nv12(input_tensor)

        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
            self.y_scale = self.x_scale # letterbox 通常 x, y 轴使用相同缩放比例

            if self.x_scale <= 0 or self.y_scale <= 0: # 确保缩放有效
                logger.warning(f"预处理中出现无效的缩放因子 ({self.x_scale}, {self.y_scale})，图像尺寸: {self.img_w}x{self.img_h}，模型输入: {self.input_W}x{self.input_H}")
                self.x_scale = 1.0
                self.y_scale = 1.0


            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift

            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift
            
            # 确保 new_w 和 new_h > 0
            if new_w <=0 or new_h <=0:
                new_w = self.img_w
                new_h = self.img_h
                self.x_shift = (self.input_W - new_w) // 2
                x_other = self.input_W - new_w - self.x_shift
                self.y_shift = (self.input_H - new_h) // 2
                y_other = self.input_H - new_h - self.y_shift
                resized_img = img # 不进行resize
            else:
                 resized_img = cv2.resize(img, (new_w, new_h))

            input_tensor = cv2.copyMakeBorder(resized_img, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            input_tensor = self.bgr2nv12(input_tensor)
        else:
            logger.error(f"非法的预处理类型 = {PREPROCESS_TYPE}")
            exit(-1)
        # logger.debug(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}, y_shift = {self.y_shift}, x_shift = {self.x_shift}")
        return input_tensor

    def bgr2nv12(self, bgr_img):
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12
        
    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"推理耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs

    def c2numpy(self, outputs):
        begin_time = time()
        outputs_np = [dnnTensor.buffer for dnnTensor in outputs] # 重命名以避免冲突
        logger.debug("\033[1;31m" + f"C 结构转 NumPy 数组耗时 = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs_np

    def postProcess(self, outputs_np):
        begin_time = time()
        s_clses = outputs_np[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs_np[1].reshape(-1, self.REG * 4)
        m_clses = outputs_np[2].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs_np[3].reshape(-1, self.REG * 4)
        l_clses = outputs_np[4].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs_np[5].reshape(-1, self.REG * 4)

        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.CONF_THRES_RAW)
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.CONF_THRES_RAW)
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.CONF_THRES_RAW)
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale
        
        s_stride, m_stride, l_stride = 8, 16, 32

        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, self.REG), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2]) * s_stride

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, self.REG), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2]) * m_stride

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, self.REG), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2]) * l_stride
        
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        # x1,y1,w,h for NMS
        xyhw2 = np.hstack([dbboxes[:,0:2], dbboxes[:,2:4] - dbboxes[:,0:2]]) 

        results = []
        # 对所有检测结果进行一次性阈值筛选
        score_mask_all = scores >= self.SCORE_THRESHOLD
        
        # 获取通过分数阈值的检测的索引、边界框、分数和类别ID
        # 注意：这里的索引是相对于 dbboxes, scores, ids 这些拼接后数组的
        indices_passed_score = np.where(score_mask_all)[0]
        
        # 如果没有检测结果通过分数阈值，则直接返回空列表
        if len(indices_passed_score) == 0:
            return results

        # 获取通过分数阈值的检测框、分数和类别
        dbboxes_passed_score = dbboxes[indices_passed_score, :]
        scores_passed_score = scores[indices_passed_score]
        ids_passed_score = ids[indices_passed_score]
        xyhw2_passed_score = xyhw2[indices_passed_score, :]

        # 对通过分数阈值的检测结果进行NMS (OpenCV NMSBoxes要求 list of bboxes 和 list of scores)
        # cv2.dnn.NMSBoxesBoxes 对每个类别是分开做的，但这里我们先对所有类别一起做，然后再按类别分开，或者，如果您想严格按类别NMS，则需要循环每个类别
        # 为了简化，这里演示一个更直接的方式，先获取所有高分框，然后NMS
        # 注意：标准的NMS通常是按类别进行的。如果模型输出很多重叠的不同类别的框，这里的全局NMS可能不是最优选择。

        for i in range(self.CLASSES_NUM):
            # 1. 筛选出当前类别的检测结果
            class_specific_mask_initial = (ids == i)
            if not np.any(class_specific_mask_initial):
                continue
            
            # 2. 对当前类别的检测结果应用分数阈值
            # class_scores_current_cat = scores[class_specific_mask_initial]
            # class_dbboxes_current_cat = dbboxes[class_specific_mask_initial, :]
            # class_xyhw2_current_cat = xyhw2[class_specific_mask_initial, :]

            # 合并类别筛选和分数筛选
            combined_mask = class_specific_mask_initial & (scores >= self.SCORE_THRESHOLD)
            if not np.any(combined_mask):
                continue
                
            current_cat_scores_for_nms = scores[combined_mask]
            current_cat_xyhw2_for_nms = xyhw2[combined_mask, :]
            current_cat_dbboxes_for_output = dbboxes[combined_mask, :] # 用于后续输出坐标

            # 3. 对筛选后的结果执行NMS
            # tolist() 是为了兼容OpenCV NMSBoxes有时对numpy数组的处理问题
            indices_after_nms = cv2.dnn.NMSBoxes(current_cat_xyhw2_for_nms.tolist(), 
                                                 current_cat_scores_for_nms.tolist(), 
                                                 self.SCORE_THRESHOLD, # NMSBoxes内部也可能用score_threshold，但我们已经筛选过了
                                                 self.NMS_THRESHOLD)

            if len(indices_after_nms) == 0:
                continue
            
            # NMSBoxes 返回的是被选中框在输入列表中的索引
            # 如果 indices_after_nms 是 (N,1) 的形状, 展平它
            if isinstance(indices_after_nms, np.ndarray) and indices_after_nms.ndim > 1:
                 indices_after_nms = indices_after_nms.flatten()

            for indic_in_nms_input in indices_after_nms:
                # indic_in_nms_input 是 current_cat_dbboxes_for_output 中的索引
                x1_model, y1_model, x2_model, y2_model = current_cat_dbboxes_for_output[indic_in_nms_input]
                final_score = current_cat_scores_for_nms[indic_in_nms_input]

                # 缩放到原始图像尺寸
                # self.x_scale, self.y_scale, self.x_shift, self.y_shift 在 preprocess 中设置
                x1 = int((x1_model - self.x_shift) / self.x_scale)
                y1 = int((y1_model - self.y_shift) / self.y_scale)
                x2 = int((x2_model - self.x_shift) / self.x_scale)
                y2 = int((y2_model - self.y_shift) / self.y_scale)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, self.img_w -1) # 确保在原始图像边界内
                y2 = min(y2, self.img_h -1) # 确保在原始图像边界内
                
                # 确保是有效的边界框
                if x1 < x2 and y1 < y2: 
                     results.append((i, final_score, x1, y1, x2, y2))
                    #  results.append((i, final_score, x1_model, y1_model, x2_model, y2_model))
        
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

rdk_colors = [ # 确保有足够的颜色，或者对 class_id 使用取模运算
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)
] * 4 # 如果类别数超过20，重复颜色列表

# 修改后的绘制函数，包含 track_id
def draw_track(img, bbox_xyxy, score, class_id, track_id) -> None:
    x1, y1, x2, y2 = map(int, bbox_xyxy) # 确保绘制时使用整数坐标
    
    # 按 track_id 着色以便于区分
    color_index = int(track_id) % len(rdk_colors) 
    # 或者按 class_id 着色: color_index = int(class_id) % len(rdk_colors)
    color = rdk_colors[color_index]

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    label = f"ID:{track_id} {coco_names[int(class_id)]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    label_y_pos = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
    # 如果可能，确保标签背景在图像边界内
    label_bg_x2 = x1 + label_width
    label_bg_y1 = label_y_pos - label_height
    label_bg_y2 = label_y_pos + (label_height // 4) # 调整以更好地适应文本

    cv2.rectangle(
        img, (x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, cv2.FILLED
    )
    cv2.putText(img, label, (x1, label_y_pos - (label_height//2) + (label_height//4) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()