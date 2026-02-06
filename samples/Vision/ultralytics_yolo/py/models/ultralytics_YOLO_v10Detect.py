# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Cauchy - WuChao

from time import time

from scipy.special import softmax
import numpy as np
import cv2

from ..runtime import RDK_YOLO_Runtime
from ..utils.rdk_yolo_logger import logger
from ..utils.plot import draw_detection
from ..utils.coco_names import coco_names
from .ultralytics_YOLO_Detect import Ultralytics_YOLO_Detect


class Ultralytics_YOLO_v10Detect(Ultralytics_YOLO_Detect):
    def __init__(self, model_path, classes_num, nms_thres, score_thres, reg, strides):
        super().__init__(model_path, classes_num, nms_thres, score_thres, reg, strides)

    def postProcess(self, outputs):
        begin_time = time()
        # reshape
        clses = [
            outputs[0].reshape(-1, self.CLASSES_NUM),
            outputs[2].reshape(-1, self.CLASSES_NUM),
            outputs[4].reshape(-1, self.CLASSES_NUM),
        ]
        bboxes = [
            outputs[1].reshape(-1, self.REG * 4),
            outputs[3].reshape(-1, self.REG * 4),
            outputs[5].reshape(-1, self.REG * 4),
        ]
        dbboxes, ids, scores = [], [], []
        for cls, bbox, stride, grid in zip(clses, bboxes, self.strides, self.grids):
            # score 筛选
            max_scores = np.max(cls, axis=1)
            bbox_selected = np.flatnonzero(max_scores >= self.CONF_THRES_RAW)
            ids.append(np.argmax(cls[bbox_selected, :], axis=1))
            # 3个Classify分类分支：Sigmoid计算
            scores.append(1 / (1 + np.exp(-max_scores[bbox_selected])))
            # dist2bbox (ltrb2xyxy)
            ltrb_selected = np.sum(
                softmax(bbox[bbox_selected, :].reshape(-1, 4, self.REG), axis=2)
                * self.weights_static,
                axis=2,
            )
            grid_selected = grid[bbox_selected, :]
            x1y1 = grid_selected - ltrb_selected[:, 0:2]
            x2y2 = grid_selected + ltrb_selected[:, 2:4]
            dbboxes.append(np.hstack([x1y1, x2y2]) * stride)
        dbboxes = np.concatenate((dbboxes), axis=0)
        scores = np.concatenate((scores), axis=0)
        ids = np.concatenate((ids), axis=0)
        hw = dbboxes[:, 2:4] - dbboxes[:, 0:2]
        xyhw2 = np.hstack([dbboxes[:, 0:2], hw])
        # 分类别nms
        results = []
        for bbox, score, class_id in zip(dbboxes, scores, ids):
            x1, y1, x2, y2 = bbox
            x1 = (x1 - self.x_shift) / self.x_scale
            y1 = (y1 - self.y_shift) / self.y_scale
            x2 = (x2 - self.x_shift) / self.x_scale
            y2 = (y2 - self.y_shift) / self.y_scale
            x1 = x1 if x1 > 0 else 0
            x2 = x2 if x2 > 0 else 0
            y1 = y1 if y1 > 0 else 0
            y2 = y2 if y2 > 0 else 0
            x1 = x1 if x1 < self.img_w else self.img_w
            x2 = x2 if x2 < self.img_w else self.img_w
            y1 = y1 if y1 < self.img_h else self.img_h
            y2 = y2 if y2 < self.img_h else self.img_h
            results.append((class_id, score, x1, y1, x2, y2))

        logger.debug(
            "\033[1;31m"
            + f"Post Process time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        return results
