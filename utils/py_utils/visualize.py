# Copyright (c) 2025 D-Robotics Corporation
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

"""
visualize: Visualization utilities for model results.

This module provides reusable helpers for rendering model outputs onto images
and logging detailed results. It includes generic drawing functions and 
specific wrappers for the YOLO26 model series.
"""

import cv2
import numpy as np
import logging
import math

# Use a consistent logger
logger = logging.getLogger("YOLO26")

# List of predefined RGB color tuples used for bounding box visualization.
rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

# Standard human pose skeleton structure (COCO format)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]

def draw_boxes(image: np.ndarray, boxes: np.ndarray, cls_ids: np.ndarray,
               scores: np.ndarray, class_names: list, colors: list) -> np.ndarray:
    """Draw bounding boxes with class names and scores."""
    for box, cls_id, score in zip(boxes, cls_ids, scores):
        x1, y1, x2, y2 = map(int, box)
        color = colors[cls_id % len(colors)]
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label = f"{name} {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(image, label, (x1, max(y1 - 5, 0)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=color, thickness=1)
    return image


def draw_masks(image: np.ndarray, boxes: np.ndarray, masks: list,
               cls_ids: list, colors: list, alpha: float = 0.3) -> None:
    """Overlay semi-transparent instance masks."""
    for class_id, box, mask in zip(cls_ids, boxes, masks):
        x1, y1, x2, y2 = map(int, box)
        if mask.size == 0 or x2 <= x1 or y2 <= y1:
            continue

        region = image[y1:y2, x1:x2]
        mask_area = mask.astype(bool)
        if not np.any(mask_area):
            continue

        color = colors[class_id % len(colors)]
        color_patch = np.zeros(region.shape, dtype=np.uint8)
        color_patch[:] = color

        region[mask_area] = (
            (1 - alpha) * region[mask_area] + alpha * color_patch[mask_area]
        ).astype(np.uint8)


def draw_rotated_boxes(img: np.ndarray, rrects: list, ids: list, scores: list,
                       class_names: list, colors: list, thickness: int = 2) -> np.ndarray:
    """Draw rotated bounding boxes (OBB)."""
    for rrect, cid, score in zip(rrects, ids, scores):
        cx, cy, w, h, a = rrect
        pts = cv2.boxPoints(((cx, cy), (w, h), a * 180 / math.pi)).astype(np.int32)
        color = colors[cid % len(colors)]
        name = class_names[cid] if cid < len(class_names) else str(cid)
        label = f"{name}: {score:.2f}"
        
        cv2.drawContours(img, [pts], 0, color, thickness)
        lx, ly = pts[0]
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (lx, ly - lh - 5), (lx + lw, ly), color, cv2.FILLED)
        cv2.putText(img, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def draw_pose(img: np.ndarray, boxes: np.ndarray, kpts: np.ndarray,
              skeleton: list = COCO_SKELETON, kpt_conf_thres: float = 0.5,
              scores: np.ndarray = None, class_ids: np.ndarray = None,
              colors: list = rdk_colors) -> np.ndarray:
    """Draw pose estimation results."""
    if scores is None: scores = np.ones(len(boxes))
    if class_ids is None: class_ids = np.zeros(len(boxes), dtype=int)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        color = colors[class_ids[i] % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Pose boxes usually green
        
        label = f"person: {scores[i]:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        for x, y, conf in kpts[i]:
            if conf >= kpt_conf_thres:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        for sk in skeleton:
            idx1, idx2 = sk[0] - 1, sk[1] - 1
            if idx1 < len(kpts[i]) and idx2 < len(kpts[i]):
                p1, p2 = kpts[i][idx1], kpts[i][idx2]
                if p1[2] >= kpt_conf_thres and p2[2] >= kpt_conf_thres:
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 1)
    return img


def draw_classification(img: np.ndarray, results: list, labels: dict,
                        pos: tuple = (10, 30), scale: float = 0.8,
                        color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw Top-K results on image."""
    for i, (cid, score) in enumerate(results):
        label_str = labels.get(cid, str(cid)) if isinstance(labels, dict) else labels[cid] if cid < len(labels) else str(cid)
        text = f"Rank {i+1}: Class {cid} ({label_str}) | Score: {score:.4f}"
        cv2.putText(img, text, (pos[0], pos[1] + i * 30), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return img

def print_detections(boxes, scores, cls_ids, class_names=None):
    """
    Print detection results in a readable format.

    Args:
        boxes: Bounding boxes (xyxy), shape (N, 4).
        scores: Confidence scores, shape (N,).
        cls_ids: Class IDs, shape (N,).
        class_names: List of class names (optional).
    """
    num_dets = len(boxes)
    print(f"\n{'='*20} Detection Report {'='*20}")
    print(f"Total Objects Found: {num_dets}")

    if num_dets == 0:
        print("No objects detected.")
        return

    # print header
    print(f"{'ID':<4} {'Label':<15} {'Score':<8} {'Box (x1, y1, x2, y2)'}")
    print("-" * 60)

    for i in range(num_dets):
        # Get class name
        cls_id = int(cls_ids[i])
        label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)

        # Format score
        score = scores[i]

        # Format box coordinates (convert to int for cleaner display)
        box = boxes[i].astype(int)

        print(f"{i:<4} {label:<15} {score:.2f}    {box.tolist()}")

    print(f"{'='*58}\n")


def print_pose_detections(boxes, scores, kpts, kpt_thres=0.5):
    """
    Print pose estimation results in a readable format.

    Args:
        boxes: Bounding boxes (xyxy), shape (N, 4).
        scores: Confidence scores, shape (N,).
        kpts: Keypoints, shape (N, 17, 3) [x, y, conf].
        kpt_thres: Threshold to count a keypoint as visible.
    """
    num_dets = len(boxes)
    print(f"\n{'='*20} Pose Detection Report {'='*20}")
    print(f"Total Persons Found: {num_dets}")
    
    if num_dets == 0:
        print("No persons detected.")
        return

    # 打印表头
    print(f"{'ID':<4} {'Score':<8} {'Box (x1, y1, x2, y2)':<25} {'Visible Kpts'}")
    print("-" * 60)

    for i in range(num_dets):
        # 统计可见关键点数量 (conf > thres)
        visible_count = np.sum(kpts[i][:, 2] > kpt_thres)
        
        # 格式化分数
        score = scores[i]
        
        # 格式化坐标
        box = boxes[i].astype(int)
        
        print(f"{i:<4} {score:.2f}    {str(box.tolist()):<25} {visible_count}/17")
    
    print(f"{'='*63}\n")


def print_obb_detections(results: list, class_names: list = None):
    """
    Print OBB detection results in a readable format.

    Args:
        results: List of dicts {'rrect': (cx, cy, w, h, a), 'score': s, 'id': id}.
        class_names: List of class names (optional).
    """
    num_dets = len(results)
    print(f"\n{'='*20} OBB Detection Report {'='*20}")
    print(f"Total Objects Found: {num_dets}")

    if num_dets == 0:
        print("No objects detected.")
        return

    # print header
    print(f"{'ID':<4} {'Label':<15} {'Score':<8} {'Angle':<8} {'Box (cx, cy, w, h)'}")
    print("-" * 75)

    for i, res in enumerate(results):
        rrect = res['rrect']
        score = res['score']
        cid = int(res['id'])
        name = class_names[cid] if class_names and cid < len(class_names) else str(cid)
        angle_deg = math.degrees(rrect[4])
        
        # Format OBB: (cx, cy, w, h)
        box_str = f"({rrect[0]:.1f}, {rrect[1]:.1f}, {rrect[2]:.1f}, {rrect[3]:.1f})"
        
        print(f"{i:<4} {name:<15} {score:.2f}    {angle_deg:>6.1f}°    {box_str}")

    print(f"{'='*75}\n")


def print_classification_results(results: list, class_names: list = None):
    """
    Print Top-K classification results.

    Args:
        results: List of (class_id, probability) tuples.
        class_names: List of class names (optional).
    """
    print(f"\n{'='*20} Classification Results {'='*20}")
    for i, (cls_id, score) in enumerate(results):
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        print(f"Top-{i+1}: Class ID {cls_id:<4} ({name:<20}) | Score: {score:.4f}")
    print(f"{'='*64}\n")


def draw_obb(img: np.ndarray, results: list, class_names: list,
             colors: list = rdk_colors) -> np.ndarray:
    """Draw OBB results and log details."""
    if not results: return img

    rrects, ids, scores = [], [], []
    for r in results:
        cid, score = int(r['id']), r['score']
        name = class_names[cid] if cid < len(class_names) else str(cid)
        logger.info(f"{name}: {score:.2f} | Angle: {r['rrect'][4] * 180 / math.pi:.1f} deg")
        rrects.append(r['rrect']); ids.append(cid); scores.append(score)

    return draw_rotated_boxes(img, rrects, ids, scores, class_names, colors)
