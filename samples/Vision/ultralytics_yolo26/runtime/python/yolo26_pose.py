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

# flake8: noqa: E501
# flake8: noqa: E402

"""Provide a YOLO26 Pose inference wrapper and pipeline utilities.

This module defines a lightweight YOLO26 Pose runtime wrapper built on 
HBM runtime. It includes configuration definitions and a complete inference 
pipeline (preprocess, forward, postprocess), aligned with the YOLOv5 standard.

Model Structure Assumption (based on provided logs):
    Inputs:
        0: images_y (1, 640, 640, 1)
        1: images_uv (1, 320, 320, 2)
    Outputs (9 tensors):
        Stride 8:  [0] Cls(1), [1] Box(4), [2] Kpt(51)
        Stride 16: [3] Cls(1), [4] Box(4), [5] Kpt(51)
        Stride 32: [6] Cls(1), [7] Box(4), [8] Kpt(51)
"""

import os
import cv2
import sys
import time
import logging
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils

logger = logging.getLogger("YOLO26_Pose")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation function.

    Args:
        x: Input NumPy array.

    Returns:
        A NumPy array with the sigmoid function applied element-wise.
    """
    return 1.0 / (1.0 + np.exp(-x))


def scale_kpts_back(kpts: np.ndarray,
                    img_w: int, img_h: int,
                    input_w: int, input_h: int,
                    resize_type: int = 1) -> np.ndarray:
    """Rescale keypoints from model input size to original image size.

    Args:
        kpts: Keypoints array shape `(N, 17, 3)`. The last dimension contains
            `[x, y, conf]`.
        img_w: Original image width.
        img_h: Original image height.
        input_w: Model input width.
        input_h: Model input height.
        resize_type: Resize strategy used during preprocessing.
            - 0: Direct resize.
            - 1: Letterbox resize.

    Returns:
        Rescaled keypoints array with shape `(N, 17, 3)`.
    """
    if resize_type == 0:
        scale_x = img_w / input_w
        scale_y = img_h / input_h
        kpts[..., 0] *= scale_x
        kpts[..., 1] *= scale_y
    elif resize_type == 1:
        scale = min(input_w / img_w, input_h / img_h)
        pad_w = (input_w - img_w * scale) / 2
        pad_h = (input_h - img_h * scale) / 2
        
        kpts[..., 0] = (kpts[..., 0] - pad_w) / scale
        kpts[..., 1] = (kpts[..., 1] - pad_h) / scale

    # Clip coordinates within valid image bounds
    kpts[..., 0] = np.clip(kpts[..., 0], 0, img_w)
    kpts[..., 1] = np.clip(kpts[..., 1], 0, img_h)
    
    return kpts


def decode_pose_layer(box_feat: np.ndarray,
                      cls_feat: np.ndarray,
                      kpt_feat: np.ndarray,
                      stride: int,
                      score_thres: float) -> np.ndarray:
    """Decode a single feature layer for Pose Estimation.

    This function decodes the raw output tensors (Box, Class, Keypoints) 
    of one detection layer. It includes an optimization to filter out 
    background anchors using raw logits before expensive sigmoid operations.

    Args:
        box_feat: Raw bounding box output tensor with shape `(1, H, W, 4)`.
            Contains distal distances.
        cls_feat: Raw classification output tensor with shape `(1, H, W, 1)`.
            Contains raw logits for the single class (Person).
        kpt_feat: Raw keypoint output tensor with shape `(1, H, W, 51)`.
            Contains 17 keypoints * 3 values (x, y, conf).
        stride: Stride of the feature layer relative to the input image.
        score_thres: Confidence threshold for pre-filtering.

    Returns:
        A NumPy array of shape `(N, 57)` containing decoded predictions, 
        formatted as `[x1, y1, x2, y2, score, cls, kpt1_x, kpt1_y, kpt1_conf...]`.
    """
    # Remove batch dimension
    if box_feat.shape[0] == 1: box_feat = box_feat[0]
    if cls_feat.shape[0] == 1: cls_feat = cls_feat[0]
    if kpt_feat.shape[0] == 1: kpt_feat = kpt_feat[0]

    h, w, _ = box_feat.shape

    # 1. Logits Filter (Optimization for Single Class)
    # Math: sigmoid(x) >= thres <==> x >= -ln(1/thres - 1)
    safe_thres = np.clip(score_thres, 1e-6, 1.0 - 1e-6)
    logit_thres = -np.log(1.0 / safe_thres - 1.0)
    
    # Since shape is (H, W, 1), we just take the 0-th channel
    raw_logits = cls_feat[..., 0]
    
    mask = raw_logits >= logit_thres
    if not np.any(mask):
        return np.empty((0, 6 + 51), dtype=np.float32)

    # 2. Select valid candidates
    grid_y, grid_x = np.indices((h, w))
    valid_grid_x = grid_x[mask]
    valid_grid_y = grid_y[mask]
    
    valid_box = box_feat[mask]       # (N, 4)
    valid_kpt = kpt_feat[mask]       # (N, 51)
    valid_logits = raw_logits[mask]  # (N,)

    # 3. Compute Scores
    valid_score = sigmoid(valid_logits)
    # Class ID is always 0 for single-class pose models
    valid_cls_id = np.zeros_like(valid_score)

    # 4. Decode Box (Distal-to-Center)
    grid_center_x = valid_grid_x.astype(np.float32) + 0.5
    grid_center_y = valid_grid_y.astype(np.float32) + 0.5
    
    x1 = (grid_center_x - valid_box[:, 0]) * stride
    y1 = (grid_center_y - valid_box[:, 1]) * stride
    x2 = (grid_center_x + valid_box[:, 2]) * stride
    y2 = (grid_center_y + valid_box[:, 3]) * stride

    # 5. Decode Keypoints
    # Reshape to (N, 17, 3) for processing
    num_kpts = valid_kpt.shape[1] // 3  # Should be 17
    valid_kpt = valid_kpt.reshape(-1, num_kpts, 3)
    
    # Kpt XY: (Raw + Grid) * Stride
    # Need to broadcast grid to (N, 1, 2)
    grid_stack = np.stack([valid_grid_x, valid_grid_y], axis=-1)[:, None, :]
    grid_stack = grid_stack.astype(np.float32) + 0.5
    
    # Standard formula for anchor-free pose: (kpt_raw + grid) * stride
    kpt_xy = (valid_kpt[..., :2] + grid_stack) * stride

    # Kpt Conf: Sigmoid(Raw)
    kpt_conf = sigmoid(valid_kpt[..., 2:3])
    
    # Concatenate back to (N, 51)
    decoded_kpt_flat = np.concatenate([kpt_xy, kpt_conf], axis=-1).reshape(-1, num_kpts * 3)

    # 6. Stack All: [x1, y1, x2, y2, score, cls, kpt_flat...]
    # Shape: (N, 6 + 51)
    out = np.stack([x1, y1, x2, y2, valid_score, valid_cls_id], axis=-1)
    out = np.concatenate([out, decoded_kpt_flat], axis=-1)
    
    return out


@dataclass
class YOLO26PoseConfig:
    """Configuration for initializing the YOLO26 Pose model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO26 pipeline.

    Attributes:
        model_path: Path to the compiled YOLO26 `.hbm` model.
        score_thres: Minimum confidence threshold.
        nms_thres: IoU threshold for NMS.
        resize_type: Image resize mode (0=stretch, 1=letterbox).
        strides: Feature map strides. Defaults to `[8, 16, 32]`.
    """
    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.65
    resize_type: int = 1
    strides: np.ndarray = field(
        default_factory=lambda: np.array([8, 16, 32], dtype=np.int32)
    )


class YOLO26Pose:
    """YOLO26 Pose estimation wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for YOLO26 Pose models,
    including input preprocessing, model execution, and postprocessing steps
    (decoding both bounding boxes and keypoints).
    """

    def __init__(self, config: YOLO26PoseConfig):
        """Initialize the YOLO26 Pose model with the given configuration.

        Args:
            config: Configuration object containing model path, preprocessing
                parameters, and postprocessing parameters.
        """
        t0 = time.time()
        self.cfg = config
        
        # Load Model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.debug(f"\033[1;31m[Pose] Load Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        
        # Infer Input Size (Assuming NHWC layout)
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters.

        Args:
            priority: Inference priority in the range [0, 255].
            bpu_cores: List of BPU core indices used for inference.

        Returns:
            None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray,
                    resize_type: Optional[int] = None,
                    image_format: Optional[str] = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into model-required tensor format.

        The input image is resized according to the specified resize strategy
        and converted from BGR format to NV12 (Y and UV planes).

        Args:
            img: Input image array.
            resize_type: Resize strategy override.
            image_format: Input image format. Currently, only `"BGR"` is
                supported.

        Returns:
            A nested input tensor dictionary in the form:
            `{model_name: {input_name: tensor}}`.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        t0 = time.time()
        
        if resize_type is None:
            resize_type = self.cfg.resize_type
        else:
            self.cfg.resize_type = resize_type
        
        # Resize and convert to NV12
        if image_format == "BGR":
            resize_img = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        else:
            raise ValueError(f"Unsupported image_format: {image_format}")

        logger.debug(f"\033[1;31m[Pose] Pre Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                `pre_process()`.

        Returns:
            A dictionary containing raw output tensors returned by the runtime.
        """
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.debug(f"\033[1;31m[Pose] Forward time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_img_w: int,
                     ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw model outputs into final pose results.

        This step includes decoding (Box + Keypoints), confidence filtering, 
        Non-Maximum Suppression (NMS), and coordinate scaling back to the 
        original image resolution.

        Args:
            outputs: Raw output tensors from inference.
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold for NMS.

        Returns:
            A tuple containing:
                - xyxy: Bounding boxes `(N, 4)` in original image coordinates.
                - score: Confidence scores `(N,)`.
                - cls: Class indices `(N,)` (Integers).
                - kpts: Keypoints `(N, 17, 3)` in original image coordinates.
        """
        t0 = time.time()
        score_thres = score_thres or self.cfg.score_thres
        nms_thres = nms_thres or self.cfg.nms_thres
        
        raw_outputs = outputs[self.model_name]
        decoded = []

        # Iterate strictly based on known output order
        # Model Output Layout:
        # Stride 8:  Indices 0 (Cls), 1 (Box), 2 (Kpt)
        # Stride 16: Indices 3 (Cls), 4 (Box), 5 (Kpt)
        # Stride 32: Indices 6 (Cls), 7 (Box), 8 (Kpt)
        
        for i, stride in enumerate(self.cfg.strides):
            base_idx = i * 3
            
            # Retrieve feature maps by name using index
            cls_name = self.output_names[base_idx]
            box_name = self.output_names[base_idx + 1]
            kpt_name = self.output_names[base_idx + 2]
            
            box_feat = raw_outputs[box_name]
            cls_feat = raw_outputs[cls_name]
            kpt_feat = raw_outputs[kpt_name]
            
            # Decode layer
            layer_pred = decode_pose_layer(box_feat, cls_feat, kpt_feat, 
                                           stride, score_thres)
            decoded.append(layer_pred)

        if not decoded:
             return np.array([]), np.array([]), np.array([]), np.array([])
        
        pred = np.concatenate(decoded, axis=0)
        
        if pred.shape[0] == 0:
             return np.array([]), np.array([]), np.array([]), np.array([])

        # Unpack predictions
        # pred shape: (N, 57) -> [x1,y1,x2,y2, score, cls, kpt(51)]
        xyxy = pred[:, :4]
        score = pred[:, 4]
        cls = pred[:, 5]
        kpts = pred[:, 6:].reshape(-1, 17, 3)

        # Step 3: Non-Maximum Suppression (NMS)
        # Using shared utils, ignores kpts for IoU calculation
        keep = post_utils.NMS(xyxy, score, cls, nms_thres)

        if not keep:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        xyxy = xyxy[keep]
        score = score[keep]
        cls = cls[keep]
        kpts = kpts[keep]

        # Step 4: Rescale Boxes to original image dimensions
        xyxy = post_utils.scale_coords_back(xyxy, ori_img_w, ori_img_h,
                                            self.input_w, self.input_h, self.cfg.resize_type)
        
        # Step 5: Rescale Keypoints to original image dimensions
        kpts = scale_kpts_back(kpts, ori_img_w, ori_img_h,
                               self.input_w, self.input_h, self.cfg.resize_type)

        logger.debug(f"\033[1;31m[Pose] Post Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        
        # Cast cls to int for type safety
        return xyxy, score, cls.astype(int), kpts

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                resize_type: Optional[int] = None,
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the complete pose estimation pipeline on a single image.

        This method internally performs preprocessing, inference, and
        postprocessing.

        Args:
            img: Input image array.
            image_format: Input image format. Currently supports `"BGR"`.
            resize_type: Resize strategy override.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold override for NMS.

        Returns:
            A tuple containing:
                - xyxy: Bounding boxes `(N, 4)`.
                - score: Confidence scores `(N,)`.
                - cls: Class indices `(N,)`.
                - kpts: Keypoints `(N, 17, 3)`.
        """
        # Original image size
        ori_img_h, ori_img_w = img.shape[:2]
        
        # 1) Preprocess
        inp = self.pre_process(img, resize_type, image_format)
        
        # 2) Inference
        out = self.forward(inp)
        
        # 3) Postprocess
        return self.post_process(out, ori_img_w, ori_img_h, score_thres, nms_thres)

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 resize_type: Optional[int] = None,
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Callable interface for the pose estimation pipeline.

        This method is functionally equivalent to calling `predict()`.

        Args:
            img: Input image array.
            image_format: Input image format.
            resize_type: Resize strategy override.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold override for NMS.

        Returns:
            Same return values as `predict()`.
        """
        return self.predict(img, image_format, resize_type, score_thres, nms_thres)