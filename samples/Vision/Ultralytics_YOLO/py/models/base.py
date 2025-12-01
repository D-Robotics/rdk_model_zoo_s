# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Cauchy - WuChao

import os
import argparse
import logging
from time import time

from tqdm import tqdm
from scipy.special import softmax
import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RDK_YOLO")


class Ultralytics_YOLO_RDK_Base:
    def __init__(self):
        self.RESIZE_TYPE = 0
        self.LETTERBOX_TYPE = 1
        self.PREPROCESS_TYPE = self.LETTERBOX_TYPE

    def __str__(self):
        return self.m.__str__()

    def __repr__(self):
        return self.m.__repr__()

    def preprocess_yuv420sp(self, img):

        logger.debug(f"PREPROCESS_TYPE = {self.PREPROCESS_TYPE}")
        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if self.PREPROCESS_TYPE == self.RESIZE_TYPE:
            # 利用resize的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            input_tensor = cv2.resize(
                img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST
            )  # 利用resize重新开辟内存节约一次
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.debug(
                "\033[1;31m"
                + f"pre process(resize) time = {1000*(time() - begin_time):.2f} ms"
                + "\033[0m"
            )
        elif self.PREPROCESS_TYPE == self.LETTERBOX_TYPE:
            # 利用 letter box 的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            self.x_scale = min(
                1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w
            )
            self.y_scale = self.x_scale
            if self.x_scale <= 0 or self.y_scale <= 0:
                raise ValueError("Invalid scale factor.")
            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift
            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift
            input_tensor = cv2.resize(img, (new_w, new_h))
            input_tensor = cv2.copyMakeBorder(
                input_tensor,
                self.y_shift,
                y_other,
                self.x_shift,
                x_other,
                cv2.BORDER_CONSTANT,
                value=[127, 127, 127],
            )
            input_tensor = self.bgr2nv12(input_tensor)
            logger.debug(
                "\033[1;31m"
                + f"pre process(letter box) time = {1000*(time() - begin_time):.2f} ms"
                + "\033[0m"
            )
        else:
            logger.error(f"illegal PREPROCESS_TYPE = {self.PREPROCESS_TYPE}")
            exit(-1)
        logger.debug(
            "\033[1;31m"
            + f"pre process time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        logger.debug(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}")
        logger.debug(f"y_shift = {self.y_shift:.2f}, x_shift = {self.x_shift:.2f}")
        return input_tensor

    def preprocess_yuv420sp_planes(self, img):
        logger.debug(f"PREPROCESS_TYPE = {self.PREPROCESS_TYPE}")
        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if self.PREPROCESS_TYPE == self.RESIZE_TYPE:
            # 利用resize的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            input_tensor = cv2.resize(
                img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST
            )  # 利用resize重新开辟内存节约一次
            y, uv = self.bgr_to_nv12_planes(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.debug(
                "\033[1;31m"
                + f"pre process(resize) time = {1000*(time() - begin_time):.2f} ms"
                + "\033[0m"
            )
        elif self.PREPROCESS_TYPE == self.LETTERBOX_TYPE:
            # 利用 letter box 的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            self.x_scale = min(
                1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w
            )
            self.y_scale = self.x_scale
            if self.x_scale <= 0 or self.y_scale <= 0:
                raise ValueError("Invalid scale factor.")
            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift
            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift
            input_tensor = cv2.resize(img, (new_w, new_h))
            input_tensor = cv2.copyMakeBorder(
                input_tensor,
                self.y_shift,
                y_other,
                self.x_shift,
                x_other,
                cv2.BORDER_CONSTANT,
                value=[127, 127, 127],
            )
            y, uv = self.bgr_to_nv12_planes(input_tensor)
            logger.debug(
                "\033[1;31m"
                + f"pre process(letter box) time = {1000*(time() - begin_time):.2f} ms"
                + "\033[0m"
            )
        else:
            logger.error(f"illegal PREPROCESS_TYPE = {self.PREPROCESS_TYPE}")
            exit(-1)
        logger.debug(
            "\033[1;31m"
            + f"pre process time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        logger.debug(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}")
        logger.debug(f"y_shift = {self.y_shift:.2f}, x_shift = {self.x_shift:.2f}")
        return y, uv

    def bgr2nv12(self, bgr_img):
        begin_time = time()
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape(
            (area * 3 // 2,)
        )
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[: height * width] = y
        nv12[height * width :] = uv_packed
        logger.debug(
            "\033[1;31m"
            + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        return nv12

    def bgr_to_nv12_planes(self, image):
        begin_time = time()
        height, width = image.shape[:2]
        area = height * width
        yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
        yuv420p = yuv420p.reshape((area * 3 // 2,))
        y = yuv420p[:area].reshape((height, width))
        u = yuv420p[area : area + area // 4].reshape((height // 2, width // 2))
        v = yuv420p[area + area // 4 :].reshape((height // 2, width // 2))
        uv = np.stack((u, v), axis=-1)
        y = y[np.newaxis, :, :, np.newaxis]
        uv = uv[np.newaxis, :, :, :]
        logger.debug(
            "\033[1;31m"
            + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        return y, uv

    def forward(self, input_tensors):
        return self.m(input_tensors)

    def end2end_forward(self, img):
        # img: HWC, BGR888, np.uint8, 0~255
        if self.m.device == "rdkx5":
            return self.postProcess(self.forward([self.preprocess_yuv420sp(img)]))
        if self.m.device == "rdks100":
            return self.postProcess(self.forward(self.preprocess_yuv420sp_planes(img)))
        return None
