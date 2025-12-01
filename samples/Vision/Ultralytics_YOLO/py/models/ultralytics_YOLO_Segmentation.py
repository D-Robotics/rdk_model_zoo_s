# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Cauchy - WuChao

from time import time

from scipy.special import softmax
import numpy as np
import cv2

from ..runtime import RDK_YOLO_Runtime
from ..utils.rdk_yolo_logger import logger
from ..utils.plot import draw_detection, rdk_colors
from ..utils.coco_names import coco_names
from .base import Ultralytics_YOLO_RDK_Base


class Ultralytics_YOLO_Segmentation(Ultralytics_YOLO_RDK_Base):
    def __init__(
        self,
        model_path,
        classes_num,
        nms_thres,
        score_thres,
        reg,
        mc,
        strides,
        is_open,
        is_point,
    ):
        super().__init__()
        self.m = RDK_YOLO_Runtime(model_path)
        self.REG = reg
        self.CLASSES_NUM = classes_num
        self.MCES_NUM = mc
        self.SCORE_THRESHOLD = score_thres
        self.NMS_THRESHOLD = nms_thres
        self.CONF_THRES_RAW = -np.log(1 / self.SCORE_THRESHOLD - 1)
        self.input_H, self.input_W = self.m.input_H, self.m.input_W
        self.strides = strides
        logger.debug(f"{self.REG = }, {self.CLASSES_NUM = }")
        logger.debug(
            "SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"
            % (self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
        )
        logger.debug("CONF_THRES_RAW = %.2f" % self.CONF_THRES_RAW)
        logger.debug(f"{self.input_H = }, {self.input_W = }")
        logger.debug(f"{self.strides = }")
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
        self.Mask_H, self.Mask_W = 160, 160
        self.x_scale_corp = self.Mask_W / self.input_W
        self.y_scale_corp = self.Mask_H / self.input_H
        logger.debug(f"{self.Mask_H = }   {self.Mask_W = }")
        logger.debug(f"{self.x_scale_corp = }, {self.y_scale_corp = }")
        self.IS_OPEN = is_open
        self.kernel_for_morphologyEx = np.ones((5, 5), np.uint8)
        logger.debug(f"{self.IS_OPEN = }   {self.kernel_for_morphologyEx = }")
        self.IS_POINT = is_point
        logger.debug(f"{self.IS_POINT = }")

    def __call__(self, img_file, img_id=0, is_draw=True, is_add=True, **kargs):
        result_str = ""
        img = cv2.imread(img_file)
        zeros = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if img is None:
            raise ValueError(f"Load image failed: {img_file}")
        id_cnt = 0
        for class_id, score, x1, y1, x2, y2, mask in self.end2end_forward(img):
            logger.debug(
                "(%d, %d, %d, %d) -> %s: %.2f"
                % (x1, y1, x2, y2, coco_names[class_id], score)
            )
            if is_draw:
                draw_detection(img, (x1, y1, x2, y2), score, class_id)
            # Instance Segment
            if mask.size == 0:
                continue
            mask = cv2.resize(
                mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_LANCZOS4
            )
            mask = (
                cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_for_morphologyEx, 1)
                if self.IS_OPEN
                else mask
            )
            zeros[y1:y2, x1:x2, :][mask == 1] = rdk_colors[(class_id - 1) % 20]
            # points
            if not self.IS_POINT:
                continue
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # 手动连接轮廓
                contour = np.vstack((contours[0], np.array([contours[0][0]])))
                for i in range(1, len(contours)):
                    contour = np.vstack(
                        (contour, contours[i], np.array([contours[i][0]]))
                    )
                # 轮廓投射回原来的图像大小
                merged_points = contour[:, 0, :]
                merged_points[:, 0] = merged_points[:, 0] + x1
                merged_points[:, 1] = merged_points[:, 1] + y1
                points = np.array(
                    [[[int(x), int(y)] for x, y in merged_points]], dtype=np.int32
                )
                # 绘制轮廓
                if is_draw:
                    cv2.polylines(
                        img,
                        points,
                        isClosed=True,
                        color=rdk_colors[(class_id - 1) % 20],
                        thickness=4,
                    )
                if len(points[0]) <= 2:
                    seg_points_str = f"{x1:.2f}\t{y1:.2f}\t{x2:.2f}\t{y1:.2f}\t{x2:.2f}\t{y2:.2f}\t{x1:.2f}\t{y2:.2f}\t"
                else:
                    seg_points_str = ""
                    for x, y in points[0]:
                        x, y = float(x), float(y)
                        seg_points_str += f"{x:.2f}\t{y:.2f}\t"
                result_str += f"{img_id}\t{id_cnt}\t{class_id}\t{score:.2f}\t{x1:.2f}\t{y1:.2f}\t{x2:.2f}\t{y2:.2f}\t{seg_points_str}\n"
            id_cnt += 1
        if is_add:
            add_result = np.clip(img + 0.3 * zeros, 0, 255).astype(np.uint8)
            result = np.hstack((img, zeros, add_result))
        else:
            result = np.hstack((img, zeros))
        result_dict = {"result_img": result, "info": result_str}
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
        mces_ = [
            outputs[2].reshape(-1, self.MCES_NUM),
            outputs[5].reshape(-1, self.MCES_NUM),
            outputs[8].reshape(-1, self.MCES_NUM),
        ]
        protos = outputs[9][0]
        dbboxes, ids, scores, mces = [], [], [], []
        for cls, bbox, mc, stride, grid in zip(
            clses, bboxes, mces_, self.strides, self.grids
        ):
            # score Select
            max_scores = np.max(cls, axis=1)
            bbox_selected = np.flatnonzero(max_scores >= self.CONF_THRES_RAW)
            ids.append(np.argmax(cls[bbox_selected, :], axis=1))
            mces.append(mc[bbox_selected, :])
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
        mces = np.concatenate((mces), axis=0)
        # xy = (dbboxes[:,2:4] + dbboxes[:,0:2])/2.0
        # hw = (dbboxes[:,2:4] - dbboxes[:,0:2])
        # xyhw = np.hstack([xy, hw])
        hw = dbboxes[:, 2:4] - dbboxes[:, 0:2]
        xyhw2 = np.hstack([dbboxes[:, 0:2], hw])
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
                # mask
                x1_corp = int(x1 * self.x_scale_corp)
                y1_corp = int(y1 * self.y_scale_corp)
                x2_corp = int(x2 * self.x_scale_corp)
                y2_corp = int(y2 * self.y_scale_corp)
                # bbox
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)
                # clip
                x1 = x1 if x1 > 0 else 0
                x2 = x2 if x2 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                y2 = y2 if y2 > 0 else 0
                x1 = x1 if x1 < self.img_w else self.img_w
                x2 = x2 if x2 < self.img_w else self.img_w
                y1 = y1 if y1 < self.img_h else self.img_h
                y2 = y2 if y2 < self.img_h else self.img_h
                # mask
                mc = mces[id_indices][indic]
                mask = (
                    np.sum(
                        mc[np.newaxis, np.newaxis, :]
                        * protos[y1_corp:y2_corp, x1_corp:x2_corp, :],
                        axis=2,
                    )
                    > 0.5
                ).astype(np.uint8)
                # append
                results.append((i, scores[id_indices][indic], x1, y1, x2, y2, mask))
        logger.debug(
            "\033[1;31m"
            + f"Post Process time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        return results
