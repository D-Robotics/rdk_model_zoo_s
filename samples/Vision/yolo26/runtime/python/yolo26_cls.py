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

"""Provide a YOLO26 Classification inference wrapper and pipeline utilities.

This module defines a lightweight YOLO26 Classification runtime wrapper built 
on HBM runtime. It includes configuration definitions and a complete inference 
pipeline (preprocess, forward, postprocess).

Model Structure Assumption (typical for BPU classification models):
    Inputs:
        0: images_y (1, 224, 224, 1) or similar res
        1: images_uv (1, 112, 112, 2)
    Outputs:
        0: softmax or logits (1, 1, 1, 1000) or (1, 1000) depending on head
"""

import os
import sys
import time
import logging
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Union

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils

logger = logging.getLogger("YOLO26_Cls")


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x.

    Args:
        x: Input logits array.

    Returns:
        Probability distribution with sum = 1.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@dataclass
class YOLO26ClsConfig:
    """Configuration for initializing the YOLO26 Classification model.

    Attributes:
        model_path: Path to the compiled `.hbm` model.
        topk: Number of top classes to return. Defaults to 5.
        resize_type: Image resize strategy.
            - 0: Stretch resize (recommended for Classification).
            - 1: Letterbox resize (Keep aspect ratio).
    """
    model_path: str
    topk: int = 5
    resize_type: int = 0


class YOLO26Cls:
    """YOLO26 Classification wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for Classification models,
    handling NV12 preprocessing, BPU inference, and Top-K postprocessing.
    """

    def __init__(self, config: YOLO26ClsConfig):
        """Initialize the YOLO26 Classification model.

        Args:
            config: Configuration object containing model path and runtime parameters.

        Raises:
            Exception: If model loading fails.
        """
        self.cfg = config
        t0 = time.time()
        
        try:
            self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
            logger.info(f"\033[1;31m[Cls] Load Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        except Exception as e:
            logger.error(f"❌ Failed to load model from {self.cfg.model_path}: {e}")
            raise e
        
        # Extract model metadata
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        # Infer input resolution (Assuming NHWC: [N, H, W, C])
        input_shape = self.input_shapes[self.input_names[0]]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters.

        Args:
            priority: Inference priority in the range [0, 255].
            bpu_cores: List of BPU core indices used for inference.
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
        """Preprocess an input image into model-required NV12 tensor format.

        Args:
            img: Input image array.
            resize_type: Resize strategy override.
            image_format: Input image format. Currently supports `"BGR"`.

        Returns:
            A nested input tensor dictionary: `{model_name: {input_name: tensor}}`.

        Raises:
            ValueError: If unsupported image format is provided.
        """
        t0 = time.time()
        
        if resize_type is None:
            resize_type = self.cfg.resize_type
        
        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        # Resize
        resized_img = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
        
        # Convert BGR to NV12 (Y and UV planes)
        y, uv = pre_utils.bgr_to_nv12_planes(resized_img)
        
        # Construct Input Feed based on Model Input signature
        # Case 1: 2 Inputs (Y plane, UV plane) - Typical for BPU
        if len(self.input_names) == 2:
            input_feed = {
                self.model_name: {
                    self.input_names[0]: y,
                    self.input_names[1]: uv
                }
            }
        # Case 2: 1 Input (NV12 combined)
        else:
            y_flat = y.flatten()
            uv_flat = uv.flatten()
            nv12_flat = np.hstack((y_flat, uv_flat))
            input_feed = {
                self.model_name: {
                    self.input_names[0]: nv12_flat.reshape(self.input_shapes[self.input_names[0]])
                }
            }
        
        logger.info(f"\033[1;31m[Cls] Pre-process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return input_feed

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary.

        Returns:
            Raw output tensor dictionary from HBM runtime.
        """
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.info(f"\033[1;31m[Cls] Forward time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return outputs

    def post_process(self, 
                     outputs: Dict[str, Dict[str, np.ndarray]], 
                     topk: Optional[int] = None) -> List[Tuple[int, float]]:
        """Process raw logits to get Top-K classification results.

        Args:
            outputs: Raw output tensors.
            topk: Number of top results to return.

        Returns:
            List of tuples `(class_id, probability)`, sorted by probability descending.
        """
        t0 = time.time()
        topk = topk or self.cfg.topk

        # Extract logits
        raw_output = outputs[self.model_name]
        # Assume output[0] is the classification logits/probs
        logits = raw_output[self.output_names[0]].reshape(-1)
        
        # Calculate Softmax
        # Math: P(y=j|x) = e^(z_j) / sum(e^(z_k))
        probs = softmax(logits)
        
        # Get Top-K indices
        # Optimization: Use argpartition for faster partial sorting if K is small and N is large, 
        # but for ImageNet (1000), argsort is fast enough.
        top_indices = np.argsort(probs)[::-1][:topk]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(probs[idx])))
        
        logger.info(f"\033[1;31m[Cls] Post Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return results

    def predict(self, 
                img: np.ndarray, 
                image_format: str = "BGR",
                resize_type: Optional[int] = None,
                topk: Optional[int] = None) -> List[Tuple[int, float]]:
        """Run the complete classification pipeline on a single image.

        Args:
            img: Input image.
            image_format: Input format (default "BGR").
            resize_type: Resize strategy override.
            topk: Top-K override.

        Returns:
            List of (class_id, probability) tuples.
        """
        # 1. Preprocess
        inp = self.pre_process(img, resize_type, image_format)
        
        # 2. Inference
        out = self.forward(inp)
        
        # 3. Postprocess
        return self.post_process(out, topk)

    def __call__(self, 
                 img: np.ndarray, 
                 image_format: str = "BGR",
                 resize_type: Optional[int] = None,
                 topk: Optional[int] = None) -> List[Tuple[int, float]]:
        """Callable interface for the classification pipeline.

        Equivalent to `predict()`.
        """
        return self.predict(img, image_format, resize_type, topk)
