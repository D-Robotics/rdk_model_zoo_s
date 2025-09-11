#!/usr/bin/env python3

import os
import argparse
import hbm_runtime
import numpy as np
import cv2
from typing import Dict, Optional


# ============== Utility Functions (extracted from utils) ==============

def bgr_to_nv12_planes(image: np.ndarray) -> tuple:
    """
    Convert a BGR image to NV12 format (Y and UV planes).
    """
    height, width = image.shape[:2]
    area = height * width

    # Convert to planar YUV I420 format
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    yuv420p = yuv420p.reshape((area * 3 // 2,))

    # Extract Y, U, V planes
    y = yuv420p[:area].reshape((height, width))
    u = yuv420p[area:area + area // 4].reshape((height // 2, width // 2))
    v = yuv420p[area + area // 4:].reshape((height // 2, width // 2))

    # Interleave U and V to form UV plane
    uv = np.stack((u, v), axis=-1)

    # Add batch and channel dimensions
    y = y[np.newaxis, :, :, np.newaxis]
    uv = uv[np.newaxis, :, :, :]

    return y, uv


def resized_image(img: np.ndarray, input_W: int, input_H: int,
                  resize_type: int = 1,
                  interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    """
    Resize image with either direct resize or letterbox strategy.
    """
    img_h, img_w = img.shape[:2]

    if resize_type == 0:  # Direct resize
        resized = cv2.resize(img, (input_W, input_H), interpolation=interpolation)
    elif resize_type == 1:  # Letterbox resize (preserve aspect ratio)
        scale = min(input_H / img_h, input_W / img_w)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        pad_w = input_W - new_w
        pad_h = input_H - new_h
        left, right = pad_w // 2, pad_w - pad_w // 2
        top, bottom = pad_h // 2, pad_h - pad_h // 2

        # Pad image with gray (127,127,127)
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=(127, 127, 127))
    else:
        raise ValueError(f"Invalid resize_type: {resize_type}, must be 0 or 1")

    return resized


def print_topk_predictions(output: np.ndarray,
                           idx2label: dict,
                           topk: int = 5) -> None:
    """
    Print top-k classification predictions.
    """
    # Softmax with stability adjustment
    exp_logits = np.exp(output - np.max(output))
    probabilities = exp_logits / np.sum(exp_logits)

    # Top-k indices
    topk_idx = np.argsort(probabilities)[-topk:][::-1]
    topk_prob = probabilities[topk_idx]

    print(f"Top-{topk} Predictions:")
    for i in range(topk):
        idx = topk_idx[i]
        prob = topk_prob[i]
        label = idx2label[idx] if idx2label and idx in idx2label else f"Class {idx}"
        print(f"{label}: {prob:.4f}")


def load_image(img_path: str) -> np.ndarray:
    """
    Load an image from file path using OpenCV.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    return img


def print_model_info(models: object) -> None:
    """Print detailed information about input and output tensors of all models."""
    print("=== Model Name List ===")
    model_names = models.model_names
    print(model_names)

    print("\n=== Model Count ===")
    print(models.model_count)

    print("\n=== Input Names ===")
    input_names = models.input_names
    for model, inputs in input_names.items():
        print(f"{model}:")
        for name in inputs:
            print(f"  - {name}")

    print("\n=== Input Tensor Shapes ===")
    input_shapes = models.input_shapes
    for model, inputs in input_shapes.items():
        print(f"{model}:")
        for name, shape in inputs.items():
            print(f"  {name} -> shape: {shape}")

    print("\n=== Output Names ===")
    output_names = models.output_names
    for model, outputs in output_names.items():
        print(f"{model}:")
        for name in outputs:
            print(f"  - {name}")

    print("\n=== Output Tensor Shapes ===")
    output_shapes = models.output_shapes
    for model, outputs in output_shapes.items():
        print(f"{model}:")
        for name, shape in outputs.items():
            print(f"  {name} -> shape: {shape}")

# ============== End Utility Functions ==============


class MobileNetV2:
    """
    @brief Wrapper class for running inference using a MobileNetV2 model through HB_HBMRuntime.
    """

    def __init__(self, opt):
        """
        @brief Initialize the MobileNetV2 model with model path and extract I/O details.

        @param opt (argparse.Namespace) Command-line or config object containing model_path.
        """
        # Load model runtime
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)

        # Retrieve model name and input/output names
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.shapes = self.model.input_shapes[self.model_name]

        # Extract input resolution (Height, Width)
        self.input_H = self.shapes[self.input_names[0]][1]
        self.input_W = self.shapes[self.input_names[0]][2]

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """
        @brief Set optional scheduling parameters such as priority and BPU core assignment.

        @param priority (int, optional) Scheduling priority (0-255).
        @param bpu_cores (list[int], optional) List of BPU core indices to use for inference.
        @return None
        """
        kwargs = {}

        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}  # Set inference priority
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}  # Assign BPU cores

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self,
                   img: np.ndarray,
                   resize_type: int = 1) -> Dict[str, Dict[str, np.ndarray]]:
        """
        @brief Preprocess input image to match model input format.

        @param img (np.ndarray) Input image in BGR format.
        @param resize_type (int) Resize method flag (default is 1).
        @return Dict[str, Dict[str, np.ndarray]]: Nested dictionary with model name and input tensors.
        """
        # Resize and convert image to NV12 format
        resize_img = resized_image(img, self.input_W, self.input_H, resize_type)
        y, uv = bgr_to_nv12_planes(resize_img)  # Extract Y and UV planes

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self,
                input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        @brief Run forward inference using the preprocessed input tensor.

        @param input_tensor (Dict[str, Dict[str, np.ndarray]]) Input data keyed by model and input name.
        @return Dict[str, np.ndarray]: Output tensors keyed by output name.
        """
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self,
                     outputs: Dict[str, np.ndarray],
                     idx2label: Dict[int, str]) -> None:
        """
        @brief Postprocess output and print top-K predicted labels.

        @param outputs (Dict[str, np.ndarray]) Output tensor dictionary from model inference.
        @param idx2label (Dict[int, str]) Mapping from class index to human-readable label.
        @return None
        """
        # Display top-K predictions from output
        print_topk_predictions(outputs[self.output_names[0]][0], idx2label)


def main() -> None:
    """
    @brief Main function to perform image classification using MobileNetV2.

    This function loads model and label files, preprocesses an input image,
    performs inference using the HB_HBMRuntime backend, and prints classification results.

    @return None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='./model/mobilenetv2_224x224_nv12.hbm',
                        help='Path to BPU Quantized *.hbm Model file.')
    parser.add_argument('--priority', type=int, default=0,
                        help="Model priority (0~255). 0 is lowest, 255 is highest.")
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="BPU core indexes to run. Provide a list of integers (e.g., --bpu_cores 0 1). ")
    parser.add_argument('--test-img', type=str,
                        default='./data/zebra_cls.jpg',
                        help='Path to load the test image.')
    parser.add_argument('--label-file', type=str,
                        default='/app/res/labels/imagenet1000_clsidx_to_labels.txt',
                        help='Path to load ImageNet label mapping file.')

    opt = parser.parse_args()

    # Download model file if not present
    if not os.path.exists(opt.model_path):
        print(f"Model file {opt.model_path} does not exist. Attempting to download...")
        
        # Get the directory containing the model path
        model_dir = os.path.dirname(opt.model_path)
        download_script = os.path.join(model_dir, 'download.sh')
        
        if os.path.exists(download_script):
            try:
                import subprocess
                print(f"Running download script: {download_script}")
                
                # Run the download script (try different shells)
                try:
                    # Try bash first (Linux/Mac/WSL)
                    result = subprocess.run(['bash', download_script], 
                                          cwd=model_dir, 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=300)  # 5 minute timeout
                except FileNotFoundError:
                    # Fallback to running wget directly if bash not available
                    try:
                        with open(download_script, 'r') as f:
                            wget_line = f.read().strip()
                        if wget_line.startswith('wget '):
                            url = wget_line.split('wget ')[1]
                            result = subprocess.run(['wget', url], 
                                                  cwd=model_dir, 
                                                  capture_output=True, 
                                                  text=True, 
                                                  timeout=300)
                        else:
                            raise Exception("Cannot parse download script")
                    except Exception:
                        # Final fallback: try to download using Python requests
                        try:
                            import requests
                            with open(download_script, 'r') as f:
                                wget_line = f.read().strip()
                            if wget_line.startswith('wget '):
                                url = wget_line.split('wget ')[1]
                                model_filename = os.path.basename(opt.model_path)
                                print(f"Downloading {url} using Python...")
                                response = requests.get(url, stream=True, timeout=60)
                                response.raise_for_status()
                                with open(opt.model_path, 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                result = type('obj', (object,), {'returncode': 0, 'stderr': ''})()
                            else:
                                raise Exception("Cannot parse download URL")
                        except Exception as req_e:
                            print(f"All download methods failed. Last error: {req_e}")
                            return
                
                if result.returncode == 0:
                    print("Download completed successfully!")
                    if not os.path.exists(opt.model_path):
                        print(f"Warning: Download script completed but model file {opt.model_path} still not found.")
                        return
                else:
                    print(f"Download failed with error: {result.stderr}")
                    return
                    
            except subprocess.TimeoutExpired:
                print("Download timed out after 5 minutes.")
                return
            except Exception as e:
                print(f"Error running download script: {e}")
                return
        else:
            print(f"Download script not found at {download_script}. Please download the model manually.")
            return

    # Load label mapping if available
    idx2label: Optional[Dict[int, str]] = None
    if os.path.exists(opt.label_file):
        with open(opt.label_file, "r") as f:
            idx2label = eval(f.read())  # Expected format: {index: label}

    # Initialize MobileNetV2 model instance
    mobilenetv2 = MobileNetV2(opt)

    # Set runtime scheduling parameters
    mobilenetv2.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model information (e.g., input/output names, shape)
    print_model_info(mobilenetv2.model)

    # Load input image from disk
    img: np.ndarray = load_image(opt.test_img)

    # Preprocess image into NV12 format
    input_array = mobilenetv2.pre_process(img)

    # Run inference on the preprocessed input
    outputs = mobilenetv2.forward(input_array)

    # Print top-K classification results
    mobilenetv2.post_process(outputs, idx2label)


if __name__ == "__main__":
    main()