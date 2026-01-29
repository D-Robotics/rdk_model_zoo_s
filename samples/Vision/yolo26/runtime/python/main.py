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

"""YOLO26 Unified Inference Entry Script.

This script supports multiple YOLO26 tasks: detect, seg, pose, cls, obb.
It loads the appropriate model wrapper based on the `--task` argument and
executes the standard pipeline.

Example:
    python main.py --task detect --model-path ... --test-img ...
    python main.py --task cls --model-path ... --test-img ...
"""

import argparse
import os
import sys
import logging
import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath("../../../../../"))

# Import local wrappers
from yolo26_det import YOLO26Detect, YOLO26Config
from yolo26_seg import YOLO26Seg, YOLO26SegConfig
from yolo26_pose import YOLO26Pose, YOLO26PoseConfig
from yolo26_cls import YOLO26Cls, YOLO26ClsConfig
from yolo26_obb import YOLO26OBB, YOLO26OBBConfig

# Import utilities
import utils.py_utils.file_io as file_io
import utils.py_utils.visualize as visualize
import utils.py_utils.inspect as inspect

# Configure logger
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser(description="YOLO26 Unified Inference")
    
    # Task & Model
    parser.add_argument('--task', type=str, required=True, choices=['detect', 'seg', 'pose', 'cls', 'obb'],
                        help="Task type: detect, seg, pose, cls, obb")
    parser.add_argument('--model-path', type=str, required=True,
                        help="Path to BPU Quantized *.hbm Model.")
    
    # Input & Output
    parser.add_argument('--test-img', type=str, default='/app/res/assets/bus.jpg',
                        help='Path to Load Test Image.')
    parser.add_argument('--label-file', type=str, default=None,
                        help='Path to label file. If None, uses default based on task.')
    parser.add_argument('--img-save-path', type=str, default='../../test_data/result.jpg',
                        help='Path to Save Result Image (except for cls).')
    
    # Hyperparameters
    parser.add_argument('--score-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.65, help='IoU threshold for NMS')
    
    # Task Specific
    parser.add_argument('--topk', type=int, default=5, help='[Cls] Top K results')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5, help='[Pose] Keypoint visibility threshold')
    parser.add_argument('--angle-sign', type=float, default=1.0, help='[OBB] Angle decoding sign multiplier')
    parser.add_argument('--angle-offset', type=float, default=0.0, help='[OBB] Angle decoding offset')

    opt = parser.parse_args()

    # 1. Check Model
    # Download logic could be added here similar to main_*.py if desired, 
    # but for a generic tool, assuming user provides valid path is safer.
    if not os.path.exists(opt.model_path):
        # Optional: Try to auto-download if standard path
        logger.warning(f"Model not found at {opt.model_path}. Please check path or download it.")
        # file_io.download_model_if_needed(opt.model_path, ...) 

    # 2. Load Image
    if not os.path.exists(opt.test_img):
        logger.error(f"Image not found: {opt.test_img}")
        return
    img = file_io.load_image(opt.test_img)

    # 3. Load Labels (Auto-select default if None)
    labels = []
    if opt.task != 'pose': # Pose doesn't strictly need a label file (person only)
        if opt.label_file and os.path.exists(opt.label_file):
             labels = file_io.load_class_names(opt.label_file)
        else:
            # Fallback defaults
            project_root = os.path.abspath("../../../../../")
            defaults = {
                'detect': 'datasets/coco/coco_classes.names',
                'seg': 'datasets/coco/coco_classes.names',
                'cls': 'datasets/imagenet/imagenet_classes.names',
                'obb': 'datasets/dotav1/dota_classes.names'
            }
            if opt.task in defaults:
                def_path = os.path.join(project_root, defaults[opt.task])
                if os.path.exists(def_path):
                    labels = file_io.load_class_names(def_path)
                    logger.info(f"Loaded default labels from {def_path}")
                else:
                    logger.warning(f"Default label file not found: {def_path}")

    # 4. Dispatch Task
    result_img = None
    
    if opt.task == 'detect':
        result_img = run_detect(opt, img, labels)
    elif opt.task == 'seg':
        result_img = run_seg(opt, img, labels)
    elif opt.task == 'pose':
        result_img = run_pose(opt, img)
    elif opt.task == 'cls':
        run_cls(opt, img, labels)
    elif opt.task == 'obb':
        result_img = run_obb(opt, img, labels)

    # 5. Save Result
    if result_img is not None:
        cv2.imwrite(opt.img_save_path, result_img)
        print(f"[Saved] Result saved to: {opt.img_save_path}")


def run_detect(opt, img, labels):
    # 1. Init
    config = YOLO26Config(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres
    )
    model = YOLO26Detect(config)
    
    # 2. Predict
    # returns boxes, scores, cls_ids
    results = model.predict(img)
    
    # 3. Visualize
    visualize.print_detections(*results, labels)
    
    image = visualize.draw_boxes(
        img, results[0], results[2], results[1], labels, visualize.rdk_colors
    )
    return image


def run_seg(opt, img, labels):
    # 1. Init
    config = YOLO26SegConfig(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres
    )
    model = YOLO26Seg(config)
    
    # 2. Predict
    # returns xyxy, scores, cls_ids, masks
    xyxy, scores, cls_ids, masks = model.predict(img)
    
    # 3. Visualize
    visualize.print_detections(xyxy, scores, cls_ids, labels)
    
    visualize.draw_masks(img, xyxy, masks, cls_ids, visualize.rdk_colors)
    image = visualize.draw_boxes(img, xyxy, cls_ids, scores, labels, visualize.rdk_colors)
    return image


def run_pose(opt, img):
    # 1. Init
    config = YOLO26PoseConfig(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres
    )
    model = YOLO26Pose(config)
    
    # 2. Predict
    # returns xyxy, scores, cls_ids, kpts
    xyxy, scores, cls_ids, kpts = model.predict(img)
    
    # 3. Visualize
    visualize.print_pose_detections(xyxy, scores, kpts, kpt_thres=opt.kpt_conf_thres)
    
    image = visualize.draw_pose(
        img, xyxy, kpts, 
        kpt_conf_thres=opt.kpt_conf_thres,
        scores=scores, class_ids=cls_ids, colors=visualize.rdk_colors
    )
    return image


def run_cls(opt, img, labels):
    # 1. Init
    config = YOLO26ClsConfig(
        model_path=opt.model_path,
        topk=opt.topk,
        resize_type=0
    )
    model = YOLO26Cls(config)
    
    # 2. Predict
    # returns list of (class_id, score)
    results = model.predict(img)
    
    # 3. Visualize (Print only)
    visualize.print_classification_results(results, labels)
    
    return None  # No image to save for classification


def run_obb(opt, img, labels):
    # 1. Init
    config = YOLO26OBBConfig(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres,
        angle_sign=opt.angle_sign,
        angle_offset=opt.angle_offset
    )
    model = YOLO26OBB(config)
    
    # 2. Predict
    # returns list of dicts
    results = model.predict(img)
    
    # 3. Visualize
    visualize.print_obb_detections(results, labels)
    
    image = visualize.draw_obb(img, results, labels, visualize.rdk_colors)
    return image


if __name__ == "__main__":
    main()
