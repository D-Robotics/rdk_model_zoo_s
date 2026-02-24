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

"""Provide a YOLO26 OBB inference wrapper and pipeline utilities.

This module defines a lightweight YOLO26 Oriented Bounding Box runtime wrapper 
built on HBM runtime. It handles rotated bounding box decoding, angle calculation,
and NMS for rotated boxes.

Model Structure Assumption (based on provided logs):
    Inputs:
        0: images_y (1, 640, 640, 1)
        1: images_uv (1, 320, 320, 2)
    Outputs (9 tensors):
        Stride 8:  [0] Cls(15), [1] Box(4), [2] Angle(1)
        Stride 16: [3] Cls(15), [4] Box(4), [5] Angle(1)
        Stride 32: [6] Cls(15), [7] Box(4), [8] Angle(1)
"""

import os
import sys
import time
import math
import logging
import hbm_runtime
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Union

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils

logger = logging.getLogger("YOLO26_OBB")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class YOLO26OBBConfig:
    """Configuration for initializing the YOLO26 OBB model.

    Attributes:
        model_path: Path to the compiled `.hbm` model.
        score_thres: Confidence threshold for filtering.
        nms_thres: IoU threshold for NMS.
        angle_sign: Multiplier for angle decoding (1.0 or -1.0).
        angle_offset: Offset in degrees to add to the decoded angle.
        regularize: Whether to regularize boxes (w > h) by swapping and rotating.
        resize_type: Resize strategy (0=Stretch, 1=Letterbox).
        strides: Feature map strides.
    """
    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.2
    angle_sign: float = 1.0
    angle_offset: float = 0.0
    regularize: bool = True
    resize_type: int = 1
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class YOLO26OBB:
    """YOLO26 Oriented Bounding Box wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for OBB models,
    handling NV12 preprocessing, BPU inference, and Rotated NMS.
    """

    def __init__(self, config: YOLO26OBBConfig):
        """Initialize the YOLO26 OBB model."""
        self.cfg = config
        
        # Precompute constants
        # Logit threshold: sigmoid(x) >= thres <==> x >= -ln(1/thres - 1)
        safe_thres = np.clip(self.cfg.score_thres, 1e-6, 1.0 - 1e-6)
        self.conf_raw = -np.log(1.0 / safe_thres - 1.0)
        self.angle_offset_rad = self.cfg.angle_offset * math.pi / 180.0
        
        t0 = time.time()
        try:
            self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
            logger.debug(f"\033[1;31m[OBB] Load Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        except Exception as e:
            logger.error(f"❌ Failed to load model from {self.cfg.model_path}: {e}")
            raise e
        
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        input_shape = self.input_shapes[self.input_names[0]]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]

        # Precompute grids for each stride
        self.grids = {}
        for s in self.cfg.strides:
            grid_h, grid_w = self.input_h // s, self.input_w // s
            # Create grid (H, W, 2) -> (x, y)
            grid = np.stack(np.indices((grid_h, grid_w))[::-1], axis=-1)
            # Shift to center of grid cell
            self.grids[s] = grid.reshape(-1, 2).astype(np.float32) + 0.5
        
        # Output mapping: {stride: (cls_idx, box_idx, angle_idx)}
        # Assuming sequential order based on typical structure
        self.map_idx = {8: (0, 1, 2), 16: (3, 4, 5), 32: (6, 7, 8)}

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters."""
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
        """Preprocess an input image into model-required NV12 tensor format."""
        t0 = time.time()
        
        if resize_type is None:
            resize_type = self.cfg.resize_type
        
        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        resized_img = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized_img)
        
        if len(self.input_names) == 2:
            input_feed = {
                self.model_name: {
                    self.input_names[0]: y,
                    self.input_names[1]: uv
                }
            }
        else:
            nv12_flat = np.hstack((y.flatten(), uv.flatten()))
            input_feed = {
                self.model_name: {
                    self.input_names[0]: nv12_flat.reshape(self.input_shapes[self.input_names[0]])
                }
            }
        
        logger.debug(f"\033[1;31m[OBB] Pre-process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return input_feed

    def forward(self, input_tensor: Dict) -> Dict:
        """Execute model inference."""
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.debug(f"\033[1;31m[OBB] Forward time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return outputs

    def post_process(self, outputs: Dict, 
                     ori_w: int, ori_h: int) -> List[Dict]:
        """Decode and post-process OBB results.

        Returns:
            List of dicts: {'rrect': (cx, cy, w, h, angle_rad), 'score': float, 'id': int}
        """
        t0 = time.time()
        raw_outputs = outputs[self.model_name]
        
        all_rrects = []
        all_scores = []
        all_cids = []

        # Iterate through strides
        for stride in self.cfg.strides:
            if stride not in self.map_idx: continue
            
            ci, bi, ai = self.map_idx[stride]
            if max(bi, ci, ai) >= len(self.output_names): continue

            box_feat = raw_outputs[self.output_names[bi]].reshape(-1, 4)
            cls_feat = raw_outputs[self.output_names[ci]]
            # Flatten H*W dimensions
            cls_feat = cls_feat.reshape(-1, cls_feat.shape[-1])
            angle_feat = raw_outputs[self.output_names[ai]].reshape(-1, 1)

            # 1. Filter by Raw Logits (Optimization)
            max_scores = np.max(cls_feat, axis=1)
            mask = max_scores >= self.conf_raw
            if not np.any(mask): continue

            # 2. Decode Valid Candidates
            v_scores_logits = max_scores[mask]
            v_scores = sigmoid(v_scores_logits)
            
            v_ids = np.argmax(cls_feat[mask], axis=1)
            v_box = np.abs(box_feat[mask]) # Ensure positive distances
            v_angle = angle_feat[mask]
            grid = self.grids[stride][mask]
            
            # 3. Decode Angle (Sigmoid -> [-0.5*pi, 0.5*pi] approx)
            # Typically angle is regressed as sigmoid(x) - 0.5 scaled by pi or pi/2
            # Here matching original logic: (sigmoid(x) - 0.5) * pi * sign + offset
            a_rad = (sigmoid(v_angle[:, 0]) - 0.5) * math.pi * self.cfg.angle_sign + self.angle_offset_rad
            
            # 4. Decode Box (Distal-to-Center Rotated)
            # v_box has [l, t, r, b]
            l, t, r, b = v_box.T
            # Center offset relative to grid center, rotated by angle 'a'
            # xf, yf are offsets in the rotated frame
            xf, yf = (r - l) / 2.0, (b - t) / 2.0
            
            c_cos, s_sin = np.cos(a_rad), np.sin(a_rad)
            
            # Rotate offsets back to image frame and add to grid center
            cx = (grid[:, 0] + xf * c_cos - yf * s_sin) * stride
            cy = (grid[:, 1] + xf * s_sin + yf * c_cos) * stride
            w = (l + r) * stride
            h = (t + b) * stride
            
            # Collect
            for _cx, _cy, _w, _h, _a, _s, _id in zip(cx, cy, w, h, a_rad, v_scores, v_ids):
                # Regularize: Ensure w >= h
                if self.cfg.regularize and _w < _h:
                    _w, _h, _a = _h, _w, _a + math.pi / 2
                
                all_rrects.append((_cx, _cy, _w, _h, _a))
                all_scores.append(float(_s))
                all_cids.append(int(_id))

        # 5. Rotated NMS
        final_res = []
        if all_rrects:
            try:
                # cv2.dnn.NMSBoxesRotated requires ((cx, cy), (w, h), angle_deg)
                # Angle should be in degrees for OpenCV
                box_list = [((r[0], r[1]), (r[2], r[3]), math.degrees(r[4])) for r in all_rrects]
                
                indices = cv2.dnn.NMSBoxesRotated(
                    box_list, all_scores, 
                    self.cfg.score_thres, self.cfg.nms_thres
                )
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        cx, cy, w, h, a_rad = all_rrects[i]
                        
                        # Scale back to original image
                        cx, cy, w, h = post_utils.scale_coords_back_obb(
                            np.array([[cx, cy, w, h]]), 
                            ori_w, ori_h, self.input_w, self.input_h, self.cfg.resize_type
                        )[0]
                        
                        final_res.append({
                            'rrect': (cx, cy, w, h, a_rad),
                            'score': all_scores[i],
                            'id': all_cids[i]
                        })
            except AttributeError:
                logger.warning("⚠️ cv2.dnn.NMSBoxesRotated not available. Skipping NMS.")
                # Fallback: Return raw top-k or just all (risky)
                pass

        logger.debug(f"\033[1;31m[OBB] Post Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return final_res

    def predict(self, 
                img: np.ndarray, 
                image_format: str = "BGR",
                resize_type: Optional[int] = None) -> List[Dict]:
        """Run OBB pipeline on a single image."""
        ori_h, ori_w = img.shape[:2]
        inp = self.pre_process(img, resize_type, image_format)
        out = self.forward(inp)
        return self.post_process(out, ori_w, ori_h)

    def __call__(self, img: np.ndarray, **kwargs) -> List[Dict]:
        return self.predict(img, **kwargs)