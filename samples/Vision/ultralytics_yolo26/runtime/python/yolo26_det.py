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

"""Provide a YOLO26 inference wrapper and pipeline utilities.

This module defines a lightweight YOLO26 (Anchor-Free) runtime wrapper built on 
HBM runtime. It includes configuration definitions and a complete inference 
pipeline (preprocess, forward, postprocess), aligned with the YOLOv5 standard.
"""

import os
import cv2
import sys
import time
import logging
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Literal

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils

logger = logging.getLogger("YOLO26")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation function.

    Args:
        x: Input NumPy array.

    Returns:
        A NumPy array with the sigmoid function applied element-wise.
    """
    return 1.0 / (1.0 + np.exp(-x))


def decode_layer(box_feat: np.ndarray,
                 cls_feat: np.ndarray,
                 stride: int,
                 score_thres: float,
                 classes_num: int = 80) -> np.ndarray:
    """Decode a single feature layer from the decoupled detection head.
    
    Optimized for HBM Runtime: Performs filtering on raw logits to avoid 
    expensive sigmoid calculations on background anchors.

    Args:
        box_feat: Raw bounding box output tensor with shape `(1, H, W, 4)`.
        cls_feat: Raw classification output tensor with shape `(1, H, W, num_classes)`.
        stride: Stride of the feature layer relative to the input image.
        score_thres: Confidence threshold for pre-filtering.
        classes_num: Number of object classes.

    Returns:
        A NumPy array of shape `(N, 6)` containing decoded predictions, 
        formatted as `[x1, y1, x2, y2, score, class_id]`.
    """
    # Remove batch dimension if present
    if box_feat.shape[0] == 1:
        box_feat = box_feat[0]  # (H, W, 4)
    if cls_feat.shape[0] == 1:
        cls_feat = cls_feat[0]  # (H, W, 80)

    h, w, _ = box_feat.shape

    # -----------------------------------------------------------
    # Optimization: Filter using Raw Logits (conf_raw)
    # Avoid calculating sigmoid/exp for thousands of background grids.
    # -----------------------------------------------------------
    
    # 1. Calculate the raw logit threshold corresponding to score_thres
    # Math: sigmoid(x) >= thres  <==>  x >= -ln(1/thres - 1)
    # We clip the threshold slightly to avoid log(0) errors if thres is extreme
    safe_thres = np.clip(score_thres, 1e-6, 1.0 - 1e-6)
    logit_thres = -np.log(1.0 / safe_thres - 1.0)

    # 2. Get max logit across classes for each anchor
    # This is much faster than calculating sigmoid for the whole tensor
    max_logits = np.max(cls_feat, axis=-1)
    
    # 3. Create mask based on raw values
    mask = max_logits >= logit_thres
    
    if not np.any(mask):
        return np.empty((0, 6), dtype=np.float32)

    # -----------------------------------------------------------
    # Decode only Valid Candidates
    # -----------------------------------------------------------

    # 4. Select valid data
    # Create coordinate grid of shape (h, w, 2)
    grid_y, grid_x = np.indices((h, w))
    # Select grid points using mask
    valid_grid_x = grid_x[mask]
    valid_grid_y = grid_y[mask]
    
    valid_box = box_feat[mask]      # (K, 4)
    valid_cls_logits = cls_feat[mask] # (K, 80)

    # 5. Apply Sigmoid ONLY to valid candidates (The speedup happens here)
    valid_cls_scores = sigmoid(valid_cls_logits)
    
    # Get final scores and class IDs
    valid_score = np.max(valid_cls_scores, axis=-1)
    valid_cls_id = np.argmax(valid_cls_scores, axis=-1)

    # 6. Decode Anchor-Free Box (Distal-to-Center)
    # box_feat contains [l, t, r, b] distances from the grid center
    # Grid needs to be shifted by 0.5 to be at center
    grid_center_x = valid_grid_x.astype(np.float32) + 0.5
    grid_center_y = valid_grid_y.astype(np.float32) + 0.5
    
    x1 = (grid_center_x - valid_box[:, 0]) * stride
    y1 = (grid_center_y - valid_box[:, 1]) * stride
    x2 = (grid_center_x + valid_box[:, 2]) * stride
    y2 = (grid_center_y + valid_box[:, 3]) * stride

    # Stack results: [x1, y1, x2, y2, score, cls]
    out = np.stack([x1, y1, x2, y2, valid_score, valid_cls_id], axis=-1)
    
    return out


def decode_outputs(output_names: list[str],
                   fp32_outputs: dict[str, np.ndarray],
                   strides: list[int],
                   score_thres: float,
                   classes_num: int = 80) -> np.ndarray:
    """Decode all feature maps from the model output.

    This function iterates over decoupled detection heads, decodes each 
    feature map using `decode_layer`, and concatenates the results.

    Assumes output order: [Box_8, Cls_8, Box_16, Cls_16, Box_32, Cls_32]

    Args:
        output_names: List of output tensor names.
        fp32_outputs: Dictionary mapping output names to FP32 NumPy arrays.
        strides: List of stride values for each detection head.
        score_thres: Confidence threshold for filtering.
        classes_num: Number of object classes.

    Returns:
        A NumPy array of shape `(N, 6)` containing all decoded predictions.
    """
    decoded = []
    
    # Iterate in pairs (Box, Class) for each stride
    # Assuming output_names are sorted or ordered as [Box0, Cls0, Box1, Cls1...]
    # Map indices: Box is 0, 2, 4... Cls is 1, 3, 5...
    for i, stride in enumerate(strides):
        box_name = output_names[i * 2]
        cls_name = output_names[i * 2 + 1]
        
        box_feat = fp32_outputs[box_name]
        cls_feat = fp32_outputs[cls_name]
        
        # Decode single layer
        layer_pred = decode_layer(box_feat, cls_feat, stride, score_thres, classes_num)
        decoded.append(layer_pred)

    if not decoded:
        return np.empty((0, 6), dtype=np.float32)
        
    return np.concatenate(decoded, axis=0)


@dataclass
class YOLO26Config:
    """Configuration for initializing the YOLO26 model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO26 pipeline.

    Attributes:
        model_path: Path to the compiled YOLO26 `.hbm` model.
        classes_num: Number of detection classes. Defaults to 80.
        resize_type: Image resize mode used during preprocessing.
            - 0: Stretch resize.
            - 1: Keep aspect ratio with padding (Letterbox).
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        strides: Feature map strides for each detection scale.
            Defaults to `[8, 16, 32]`.
    """
    model_path: str
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: float = 0.45
    # Feature map downsampling strides
    strides: np.ndarray = field(
        default_factory=lambda: np.array([8, 16, 32], dtype=np.int32)
    )


class YOLO26Detect:
    """YOLO26 object detection wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for YOLO26 (Anchor-Free),
    including input preprocessing, model execution, and postprocessing steps.
    Structure matches the YOLOv5X reference implementation.
    """

    def __init__(self, config: YOLO26Config):
        """Initialize the YOLO26 model with the given configuration.

        Args:
            config: Configuration object containing model path, preprocessing
                parameters, and postprocessing parameters.
        """
        t0 = time.time()
        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)
        logger.debug(f"\033[1;31m[Detect] Load Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Model input resolution (H, W) inferred from model input tensor
        # Assuming NHWC layout
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

        # Detection and preprocessing configuration
        self.cfg = config

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
        
        Note: logic is identical to YOLOv5X implementation.

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
        
        logger.debug(f"\033[1;31m[Detect] Pre Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")

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
        logger.debug(f"\033[1;31m[Detect] Forward time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_img_w: int,
                     ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw model outputs into final detection results.

        This step includes dequantization (if needed), decoding, confidence 
        filtering, Non-Maximum Suppression (NMS), and coordinate scaling back 
        to the original image resolution.

        Args:
            outputs: Raw output tensors from inference.
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold for NMS.

        Returns:
            A tuple containing:
                - xyxy: Bounding boxes with shape `(N, 4)` in original image
                  coordinates.
                - score: Confidence scores with shape `(N,)`.
                - cls: Class indices with shape `(N,)`.
        """
        t0 = time.time()
        score_thres = score_thres or self.cfg.score_thres
        nms_thres = nms_thres or self.cfg.nms_thres

        # Step 1: Extract outputs (assuming FP32 or handling dequant internally if needed)
        # Note: If model output is int8, use post_utils.dequantize_outputs
        # For this implementation, we assume runtime returns standard dict.
        raw_outputs = outputs[self.model_name]
        
        # Step 2: Decode YOLO26 outputs into unified predictions
        # pred shape: (N, 6) -> [x1, y1, x2, y2, score, cls]
        pred = decode_outputs(self.output_names, raw_outputs,
                              self.cfg.strides, score_thres, self.cfg.classes_num)

        if pred.shape[0] == 0:
             return np.array([]), np.array([]), np.array([])

        xyxy_boxes = pred[:, :4]
        score = pred[:, 4]
        cls = pred[:, 5]

        # Step 3: Non-Maximum Suppression (NMS)
        # using shared utils.NMS
        keep = post_utils.NMS(xyxy_boxes, score, cls, nms_thres)

        if not keep:
            return np.array([]), np.array([]), np.array([])
            
        xyxy_boxes = xyxy_boxes[keep]
        score = score[keep]
        cls = cls[keep].astype(int)

        # Step 4: Rescale boxes to original image dimensions
        xyxy = post_utils.scale_coords_back(xyxy_boxes, ori_img_w, ori_img_h,
                                            self.input_w, self.input_h, self.cfg.resize_type)

        logger.debug(f"\033[1;31m[Detect] Post Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        return xyxy, score, cls

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                resize_type: Optional[int] = None,
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the complete detection pipeline on a single image.

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
                - xyxy: Bounding boxes with shape `(N, 4)`.
                - score: Confidence scores with shape `(N,)`.
                - cls: Class indices with shape `(N,)`.
        """
        # Original image size
        ori_img_h, ori_img_w = img.shape[:2]

        # 1) Preprocess
        input_tensor = self.pre_process(img, resize_type, image_format)

        # 2) Inference
        outputs = self.forward(input_tensor)

        # 3) Postprocess
        xyxy, score, cls = self.post_process(outputs, ori_img_w, ori_img_h,
                                             score_thres, nms_thres)

        return xyxy, score, cls

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 resize_type: Optional[int] = None,
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Callable interface for the detection pipeline.

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