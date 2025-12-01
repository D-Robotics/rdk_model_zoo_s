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
from .base import Ultralytics_YOLO_RDK_Base


class Ultralytics_YOLO_Pose(Ultralytics_YOLO_RDK_Base):
    def __init__(
        self,
        model_path,
        classes_num,
        nms_thres,
        score_thres,
        reg,
        strides,
        nkpt,
        kpt_conf_thres,
    ):
        super().__init__()
        self.m = RDK_YOLO_Runtime(model_path)
        self.REG = reg
        self.CLASSES_NUM = classes_num
        self.SCORE_THRESHOLD = score_thres
        self.NMS_THRESHOLD = nms_thres
        self.CONF_THRES_RAW = -np.log(1 / self.SCORE_THRESHOLD - 1)
        self.input_H, self.input_W = self.m.input_H, self.m.input_W
        self.strides = strides
        self.nkpt = nkpt
        self.kpt_conf_thres = kpt_conf_thres
        logger.debug(f"{self.REG = }, {self.CLASSES_NUM = }")
        logger.debug(
            "SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"
            % (self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
        )
        logger.debug("CONF_THRES_RAW = %.2f" % self.CONF_THRES_RAW)
        logger.debug(f"{self.input_H = }, {self.input_W = }")
        logger.debug(f"{self.strides = }")
        logger.debug(f"{self.nkpt = }")
        self.weights_static = np.array([i for i in range(reg)]).astype(np.float32)[
            np.newaxis, np.newaxis, :
        ]
        logger.debug(f"{self.weights_static.shape = }")
        self.grids = []
        for stride in self.strides:
            assert (
                self.input_H % stride == 0
            ), f"{stride=}, {self.input_H=}: input_H % stride != 0"
            assert (
                self.input_W % stride == 0
            ), f"{stride=}, {self.input_W=}: input_W % stride != 0"
            grid_H, grid_W = self.input_H // stride, self.input_W // stride
            self.grids.append(
                np.stack(
                    [
                        np.tile(np.linspace(0.5, grid_H - 0.5, grid_H), reps=grid_H),
                        np.repeat(np.arange(0.5, grid_W + 0.5, 1), grid_W),
                    ],
                    axis=0,
                ).transpose(1, 0)
            )
            logger.debug(f"{self.grids[-1].shape = }")

    def __call__(self, img_file, img_id=0, is_draw=True, **kargs):
        result_str = ""
        img = cv2.imread(img_file)
        if img is None:
            raise ValueError(f"Load image failed: {img_file}")
        id_cnt = 0
        kpt_conf_inverse = -np.log(1 / self.kpt_conf_thres - 1)
        for class_id, score, x1, y1, x2, y2, kpts in self.end2end_forward(img):
            logger.debug(
                "(%d, %d, %d, %d) -> %s: %.2f"
                % (x1, y1, x2, y2, coco_names[class_id], score)
            )
            if is_draw:
                draw_detection(img, (x1, y1, x2, y2), score, class_id)
                for j in range(self.nkpt):
                    if kpts[j][2] > kpt_conf_inverse:
                        x, y = int(kpts[j][0]), int(kpts[j][1])
                        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                        cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
                        cv2.putText(
                            img,
                            "%d" % j,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            3,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            "%d" % j,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
            kpts_txt = ""
            for kpt_cnt in range(self.nkpt):
                kpt_x, kpt_y, kpt_score = (
                    int(kpts[kpt_cnt][0]),
                    int(kpts[kpt_cnt][1]),
                    kpts[kpt_cnt][2],
                )
                if kpt_score < kpt_conf_inverse:
                    vis = 2
                else:
                    vis = 1
                kpts_txt += f"{kpt_x:.2f}\t{kpt_y:.2f}\t{int(vis)}\t"
            result_str += f"{img_id}\t{id_cnt}\t{class_id}\t{score:.2f}\t{x1:.2f}\t{y1:.2f}\t{x2:.2f}\t{y2:.2f}\t{kpts_txt}\n"
        result_dict = {"result_img": img, "info": result_str}
        return result_dict

    def postProcess(self, outputs):
        begin_time = time()
        # reshape
        clses = [
            outputs[0].reshape(-1, self.CLASSES_NUM),
            outputs[3].reshape(-1, self.CLASSES_NUM),
            outputs[6].reshape(-1, self.CLASSES_NUM),
        ]
        bboxes = [
            outputs[1].reshape(-1, self.REG * 4),
            outputs[4].reshape(-1, self.REG * 4),
            outputs[7].reshape(-1, self.REG * 4),
        ]
        kpts = [
            outputs[2].reshape(-1, self.nkpt * 3),
            outputs[5].reshape(-1, self.nkpt * 3),
            outputs[8].reshape(-1, self.nkpt * 3),
        ]
        dbboxes, ids, scores, kpts_xy, kpts_score = [], [], [], [], []
        for cls, bbox, stride, grid, kpt in zip(
            clses, bboxes, self.strides, self.grids, kpts
        ):
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
            # kpts
            kpt = kpt[bbox_selected, :].reshape(-1, 17, 3)
            kpts_xy.append(
                (kpt[:, :, :2] * 2.0 + (grid[bbox_selected, :][:, np.newaxis, :] - 0.5))
                * stride
            )
            kpts_score.append(kpt[:, :, 2:3])
        dbboxes = np.concatenate((dbboxes), axis=0)
        scores = np.concatenate((scores), axis=0)
        ids = np.concatenate((ids), axis=0)
        hw = dbboxes[:, 2:4] - dbboxes[:, 0:2]
        xyhw2 = np.hstack([dbboxes[:, 0:2], hw])
        kpts_xy = np.concatenate((kpts_xy), axis=0)
        kpts_score = np.concatenate((kpts_score), axis=0)
        # 分类别nms
        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids == i
            indices = cv2.dnn.NMSBoxes(
                xyhw2[id_indices, :],
                scores[id_indices],
                self.SCORE_THRESHOLD,
                self.NMS_THRESHOLD,
            )
            if len(indices) == 0:
                continue
            for indic in indices:
                x1, y1, x2, y2 = dbboxes[id_indices, :][indic]
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)
                x1 = x1 if x1 > 0 else 0
                x2 = x2 if x2 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                y2 = y2 if y2 > 0 else 0
                x1 = x1 if x1 < self.img_w else self.img_w
                x2 = x2 if x2 < self.img_w else self.img_w
                y1 = y1 if y1 < self.img_h else self.img_h
                y2 = y2 if y2 < self.img_h else self.img_h
                kpts_ = []
                for j in range(self.nkpt):
                    kpt_x = kpts_xy[id_indices, :][indic][j, 0]
                    kpt_y = kpts_xy[id_indices, :][indic][j, 1]
                    kpt_score = kpts_score[id_indices, :][indic][j, 0]
                    kpt_x = int((kpt_x - self.x_shift) / self.x_scale)
                    kpt_y = int((kpt_y - self.y_shift) / self.y_scale)
                    kpt_x = kpt_x if kpt_x > 0 else 0
                    kpt_y = kpt_y if kpt_y > 0 else 0
                    kpt_x = kpt_x if kpt_x < self.img_w else self.img_w
                    kpt_y = kpt_y if kpt_y < self.img_h else self.img_h
                    kpts_.append((kpt_x, kpt_y, kpt_score))
                results.append((i, scores[id_indices][indic], x1, y1, x2, y2, kpts_))
        logger.debug(
            "\033[1;31m"
            + f"Post Process time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        return results
