# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Cauchy - WuChao

import os
import argparse
import logging
from time import time

# tqdm
try:
    from tqdm import tqdm
except:
    os.system("pip install tqdm")
    from tqdm import tqdm

# scipy
try:
    from scipy.special import softmax
except:
    os.system("pip install scipy")
    from scipy.special import softmax

# numpy
try:
    import numpy as np
except:
    os.system("pip install numpy==1.26.4")
    import numpy as np

# opencv-python
try:
    import cv2
except:
    os.system("pip install opencv-python==4.10.0.84")
    import cv2


from .utils.rdk_yolo_logger import logger
from .models import (
    Ultralytics_YOLO_Detect,
    Ultralytics_YOLO_Segmentation,
    Ultralytics_YOLO_Pose,
    Ultralytics_YOLO_v10Detect,
    Ultralytics_YOLO_Classification,
)
from .runtime import DeviceSelect


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    # program configs
    parser.add_argument("--model-path", type=str, default="BPU/Nash-e/yolo11n_seg_nashe_640x640_nv12.hbm", help="Path to BPU Model.", )
    parser.add_argument("--source", type=str, default="../../../resource/datasets/COCO2017/assets/", help="image file or images path.", )
    parser.add_argument("--workspace", type=str, default="result_workspace", help="workspace path.", )
    parser.add_argument("--mode", type=str, default="default", help="default / coco2017 / imagenet1k", )
    parser.add_argument("--mode-save-name", type=str, default="result_bpu.txt", help="", )
    parser.add_argument("--yolo-type", type=str, default="yolo11", help=f"yolov5u, yolov8, yolov9, yolov10, yolo11, yolo12", )
    parser.add_argument( "--model-type", type=str, default="detect", help=f"detect, segmentation, pose, classification", )
    # model configs, Detect
    parser.add_argument("--classes-num", type=int, default=80, help="Classes Num to Detect.", )
    parser.add_argument("--nms-thres", type=float, default=0.7, help="IoU threshold, default 0.7.", )
    parser.add_argument("--score-thres", type=float, default=0.25, help="confidence threshold.", )
    parser.add_argument("--reg", type=int, default=16, help="DFL reg layer, default 16, sometimes 26.", )
    parser.add_argument("--strides", type=lambda s: list(map(int, s.split(","))), default=[8, 16, 32], help="--strides 8, 16, 32", )
    # Segmentation
    parser.add_argument("--mc", type=int, default=32, help="Mask Coefficients, default 32.", )
    parser.add_argument("--is-open", type=bool, default=True, help="Ture: morphologyEx, default True for better viewer.", )
    parser.add_argument("--is-point", type=bool, default=False, help="Ture: Draw edge points, default False, true for edge points.", )
    # Pose
    parser.add_argument("--pose-classes-num", type=int, default=1, help="Classes Num to Detect, default 1.", )
    parser.add_argument("--nkpt", type=int, default=17, help="num of keypoints, default17.", )
    parser.add_argument("--kpt-conf-thres", type=float, default=0.5, help="confidence threshold.", )
    # fmt: off
    opt = parser.parse_args()
    logger.info(opt)
    # check model file
    if not os.path.exists(opt.model_path):
        opt.model_path = download_default_model(opt)    
    # init image file or path
    source, names = make_source_images(opt)
    # make model
    m = make_model(opt)
    # workspace
    os.makedirs(opt.workspace, exist_ok=True)
    if opt.mode == "coco2017":
        coco2017_mode(opt, m, names, source)
    elif opt.mode == "imagenet1k":
        imagenet1k_mode(opt, m, names, source)
    else:
        # default
        for cnt in tqdm(range(len(names)), desc=f"inference: ", unit=" item", ncols=70):
            name = names[cnt]
            image_path = os.path.join(source, name)
            result = m(image_path)
            cv2.imwrite(
                os.path.join(opt.workspace, f"{name}_result.jpg"), result["result_img"]
            )


def imagenet1k_mode(opt, m, names, source):
    if len(names) == 50000:
        logger.info(f"mode: {opt.mode}, len(source)==50000, check success.")
        result_str = ""
        for cnt in tqdm(range(len(names)), desc=f"inference: ", unit=" item", ncols=70):
            name = names[cnt]
            image_path = os.path.join(source, name)
            result = m(image_path, name=name)
            result_str += result["info"]
        with open(
            os.path.join(opt.workspace, opt.mode_save_name), "w", encoding="utf-8"
        ) as file:
            file.write(result_str)
    else:
        logger.error(
            f"mode: {opt.mode}, len(source)={len(names)}!=50000, check falied!"
        )
        exit(1)


def coco2017_mode(opt, m, names, source):
    if len(names) == 5000:
        logger.info(f"mode: {opt.mode}, len(source)==5000, check success.")
        if opt.model_type == "segmentation":
            m.IS_POINT = True
        result_str = ""
        for cnt in tqdm(range(len(names)), desc=f"inference: ", unit=" item", ncols=70):
            name = names[cnt]
            image_path = os.path.join(source, name)
            result = m(image_path, img_id=int(name[:-4]), is_add=False, draw=False)
            result_str += result["info"]
        with open(
            os.path.join(opt.workspace, opt.mode_save_name), "w", encoding="utf-8"
        ) as file:
            file.write(result_str)
    else:
        logger.error(f"mode: {opt.mode}, len(source)={len(names)}!=5000, check falied!")
        exit(1)


def make_source_images(opt):
    # make dictonary or file to image name list
    if os.path.isfile(opt.source):
        source = os.path.dirname(opt.source)
        names = [
            name
            for name in [os.path.basename(opt.source)]
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    elif os.path.isdir(opt.source):
        source = opt.source
        names = [
            name
            for name in os.listdir(opt.source)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    else:
        logger.error(f"{opt.source}: No such file or directory")
        exit(1)
    if len(names) == 0:
        logger.error(f"source has no file endswith: png, jpg, jpeg")
        exit(1)
    return source, names


def download_default_model(opt):
    # get device
    device = DeviceSelect()
    # download default BPU model
    if opt.model_type == "detect":
        if opt.yolo_type in ["yolo26"]:
            m = None
        elif opt.yolo_type in ["yolov5u", "yolov8", "yolov9", "yolo11", "yolo12"]:
            url = {
                "rdkx5": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8/Bayes-e/yolo11n_detect_bayese_640x640_nv12.bin",
                "rdks100": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-e/yolo11n_detect_nashe_640x640_nv12.hbm",
                "rdks100p": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-m/yolo11n_detect_nashm_640x640_nv12.hbm",
            }[device()]
            os.system(f"wget -c {url}")
            return {
                "rdkx5": "yolo11n_detect_bayese_640x640_nv12.bin",
                "rdks100": "yolo11n_detect_nashe_640x640_nv12.hbm",
                "rdks100p": "yolo11n_detect_nashm_640x640_nv12.hbm",
            }[device()]
        elif opt.yolo_type in ["yolov10"]:
            url = {
                "rdkx5": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8/Bayes-e/yolov10n_detect_bayese_640x640_nv12.bin",
                "rdks100": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-e/yolov10n_detect_nashe_640x640_nv12.hbm",
                "rdks100p": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-m/yolov10n_detect_nashm_640x640_nv12.hbm",
            }[device()]
            os.system(f"wget -c {url}")
            return {
                "rdkx5": "yolov10n_detect_bayese_640x640_nv12.bin",
                "rdks100": "yolov10n_detect_nashe_640x640_nv12.hbm",
                "rdks100p": "yolov10n_detect_nashm_640x640_nv12.hbm",
            }[device()]
        else:
            m = None
    elif opt.model_type == "segmentation":
        if opt.yolo_type in ["yolo26"]:
            m = None
        elif opt.yolo_type in ["yolov8", "yolov9", "yolo11"]:
            url = {
                "rdkx5": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8/Bayes-e/yolo11n_seg_bayese_640x640_nv12.bin",
                "rdks100": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-e/yolo11n_seg_nashe_640x640_nv12.hbm",
                "rdks100p": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-m/yolo11n_seg_nashm_640x640_nv12.hbm",
            }[device()]
            os.system(f"wget -c {url}")
            return {
                "rdkx5": "yolo11n_seg_bayese_640x640_nv12.bin",
                "rdks100": "yolo11n_seg_nashe_640x640_nv12.hbm",
                "rdks100p": "yolo11n_seg_nashm_640x640_nv12.hbm",
            }[device()]
        else:
            m = None
    elif opt.model_type == "pose":
        if opt.yolo_type in ["yolo26"]:
            m = None
        elif opt.yolo_type in ["yolov8", "yolo11"]:
            url = {
                "rdkx5": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8/Bayes-e/yolo11n_pose_bayese_640x640_nv12.bin",
                "rdks100": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-e/yolo11n_pose_nashe_640x640_nv12.hbm",
                "rdks100p": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-m/yolo11n_pose_nashm_640x640_nv12.hbm",
            }[device()]
            os.system(f"wget -c {url}")
            return {
                "rdkx5": "yolo11n_pose_bayese_640x640_nv12.bin",
                "rdks100": "yolo11n_pose_nashe_640x640_nv12.hbm",
                "rdks100p": "yolo11n_pose_nashm_640x640_nv12.hbm",
            }[device()]
        else:
            m = None
    elif opt.model_type == "classification":
        if opt.yolo_type in ["yolo26"]:
            m = None
        elif opt.yolo_type in ["yolov8", "yolo11"]:
            url = {
                "rdkx5": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8/Bayes-e/yolo11n_cls_bayese_640x640_nv12.bin",
                "rdks100": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-e/yolo11s_cls_nashe_640x640_nv12.hbm",
                "rdks100p": "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.5.0/Nash-m/yolo11n_cls_nashm_640x640_nv12.hbm",
            }[device()]
            os.system(f"wget -c {url}")
            return {
                "rdkx5": "yolo11n_cls_bayese_640x640_nv12.bin",
                "rdks100": "yolo11s_cls_nashe_640x640_nv12.hbm",
                "rdks100p": "yolo11n_cls_nashm_640x640_nv12.hbm",
            }[device()]
        else:
            m = None
    else:
        m = None
    if m is not None:
        logger.info(
            f"YOLO Type: {opt.yolo_type}, Model Type: {opt.model_type}, init success."
        )
        logger.debug(m)
    else:
        logger.error(
            f"YOLO Type: {opt.yolo_type}, Model Type: {opt.model_type}, not supported."
        )
        exit(1)
    return m


def make_model(opt):
    # init model by yolo type and model type
    if opt.model_type == "detect":
        if opt.yolo_type in ["yolo26"]:
            m = None
        elif opt.yolo_type in ["yolov5u", "yolov8", "yolov9", "yolo11", "yolo12"]:
            m = Ultralytics_YOLO_Detect(
                model_path=opt.model_path,
                classes_num=opt.classes_num,  # default: 80
                nms_thres=opt.nms_thres,  # default: 0.7
                score_thres=opt.score_thres,  # default: 0.25
                reg=opt.reg,  # default: 16
                strides=opt.strides,  # default: [8, 16, 32]
            )
        elif opt.yolo_type in ["yolov10"]:
            m = Ultralytics_YOLO_v10Detect(
                model_path=opt.model_path,
                classes_num=opt.classes_num,  # default: 80
                nms_thres=opt.nms_thres,  # default: 0.7
                score_thres=opt.score_thres,  # default: 0.25
                reg=opt.reg,  # default: 16
                strides=opt.strides,  # default: [8, 16, 32]
            )
        else:
            m = None
    elif opt.model_type == "segmentation":
        if opt.yolo_type in ["yolo26"]:
            m = None
        elif opt.yolo_type in ["yolov8", "yolov9", "yolo11"]:
            m = Ultralytics_YOLO_Segmentation(
                model_path=opt.model_path,
                classes_num=opt.classes_num,  # default: 80
                nms_thres=opt.nms_thres,  # default: 0.7
                score_thres=opt.score_thres,  # default: 0.25
                reg=opt.reg,  # default: 16
                mc=opt.mc,  # default: 32
                strides=opt.strides,  # default: [8, 16, 32]
                is_open=opt.is_open,  # default: False
                is_point=opt.is_point,  # default: False
            )
        else:
            m = None
    elif opt.model_type == "pose":
        if opt.yolo_type in ["yolo26"]:
            m = None
        elif opt.yolo_type in ["yolov8", "yolo11"]:
            m = Ultralytics_YOLO_Pose(
                model_path=opt.model_path,
                classes_num=opt.pose_classes_num,  # default: 1
                nms_thres=opt.nms_thres,  # default: 0.7
                score_thres=opt.score_thres,  # default: 0.25
                reg=opt.reg,  # default: 16
                strides=opt.strides,  # default: [8, 16, 32]
                nkpt=opt.nkpt,  # default: 17
                kpt_conf_thres=opt.kpt_conf_thres,  # default: 0.5
            )
        else:
            m = None
    elif opt.model_type == "classification":
        if opt.yolo_type in ["yolo26"]:
            m = None
        elif opt.yolo_type in ["yolov8", "yolo11"]:
            m = Ultralytics_YOLO_Classification(model_path=opt.model_path)
        else:
            m = None
    else:
        m = None
    if m is not None:
        logger.info(
            f"YOLO Type: {opt.yolo_type}, Model Type: {opt.model_type}, init success."
        )
        logger.debug(m)
    else:
        logger.error(
            f"YOLO Type: {opt.yolo_type}, Model Type: {opt.model_type}, not supported."
        )
        exit(1)
    return m


if __name__ == "__main__":
    main()
