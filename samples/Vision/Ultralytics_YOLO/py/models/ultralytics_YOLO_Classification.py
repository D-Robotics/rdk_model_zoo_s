# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Cauchy - WuChao

from time import time

from scipy.special import softmax
import numpy as np
import cv2

from ..runtime import RDK_YOLO_Runtime
from ..utils.rdk_yolo_logger import logger
from ..utils.plot import rdk_colors
from ..utils.imagenet1k_names import IMAGENET2012_CLASSES
from .base import Ultralytics_YOLO_RDK_Base


class Ultralytics_YOLO_Classification(Ultralytics_YOLO_RDK_Base):
    def __init__(self, model_path):
        super().__init__()
        self.m = RDK_YOLO_Runtime(model_path)
        self.input_H, self.input_W = self.m.input_H, self.m.input_W
        self.PREPROCESS_TYPE = self.RESIZE_TYPE

    def __call__(self, img_file, is_draw=True, name="", **kargs):
        result_str = ""
        img = cv2.imread(img_file)
        if img is None:
            raise ValueError(f"Load image failed: {img_file}")
        id_, score = self.end2end_forward(img)
        if is_draw:
            y_begin = 0
            for i in range(5):
                label = f"TOP{i+1}: {id_[i]}, {100*score[i]:.1f}%, name: {list(IMAGENET2012_CLASSES.values())[id_[i]]}"
                color = rdk_colors[id_[i] % 20]
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                label_width, label_height = label_width + 2, label_height + 2
                label_x, label_y = 0, y_begin
                y_begin += label_height + 2
                cv2.rectangle(
                    img,
                    (label_x, label_y),
                    (label_x + label_width, label_y + label_height),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    img,
                    label,
                    (label_x, label_y + label_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
        result_str += (
            f"{name}\t{id_[0]}\t{id_[0]}\t{id_[1]}\t{id_[2]}\t{id_[3]}\t{id_[4]}\n"
        )
        result_dict = {"result_img": img, "info": result_str}
        return result_dict

    def postProcess(self, outputs):
        begin_time = time()
        scores = softmax(outputs[0].reshape(-1))
        top5_ids = np.argsort(scores)[-5:][::-1]  # argsort默认升序，[::-1]反转为降序
        top5_scores = scores[top5_ids]
        logger.debug(
            "\033[1;31m"
            + f"Post Process time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        return top5_ids, top5_scores
