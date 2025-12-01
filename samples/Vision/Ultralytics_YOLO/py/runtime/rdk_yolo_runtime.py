# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Cauchy - WuChao


import os
from time import time

import numpy as np

from ..utils.rdk_yolo_logger import logger
from .rdk_device_select import DeviceSelect

DEVICE_TREE_PATH = "/sys/firmware/devicetree/base/model"
RDK_X5_PIP_INSTALL_COMMAND = "pip install hobot-dnn-rdkx5"
RDK_S100_PIP_INSTALL_COMMAND = "pip install hbm-runtime"


class RDK_YOLO_Runtime:
    def __init__(self, model_path):
        self.deviceSelect = DeviceSelect()
        if self.deviceSelect() in ["rdkx5", 'rdkx5m']:
            try:
                try:
                    from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
                except:
                    from hobot_dnn_rdkx5 import (
                        pyeasy_dnn as dnn,
                    )  # BSP Python API from PyPI
            except:
                os.system(RDK_X5_PIP_INSTALL_COMMAND)
                from hobot_dnn_rdkx5 import pyeasy_dnn as dnn
            self.device = "rdkx5"
            logger.info("\033[31m" + "Auto Select Device: RDK X5" + "\033[0m")
            try:
                begin_time = time()
                self.m = dnn.load(model_path)
                logger.debug(
                    "\033[1;31m"
                    + "Load D-Robotics BPU model time = %.2f ms"
                    % (1000 * (time() - begin_time))
                    + "\033[0m"
                )
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load model file: {model_path}, {e}")
            self.input_H, self.input_W = self.m[0].inputs[0].properties.shape[2:4]
        elif self.deviceSelect() in ["rdks100", 'rdks100p']:
            try:
                from hbm_runtime import HB_HBMRuntime
            except:
                os.system(RDK_S100_PIP_INSTALL_COMMAND)
                from hbm_runtime import HB_HBMRuntime
            self.device = "rdks100"
            logger.info(
                "\033[31m" + "Auto Select Device: RDK S100 / RDK S100P" + "\033[0m"
            )
            try:
                begin_time = time()
                self.m = HB_HBMRuntime(model_path)
                logger.debug(
                    "\033[1;31m"
                    + "Load D-Robotics BPU model time = %.2f ms"
                    % (1000 * (time() - begin_time))
                    + "\033[0m"
                )
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load model file: {model_path}, {e}")
            self.input_H, self.input_W = self.m.input_shapes[self.m.model_names[0]][
                self.m.input_names[self.m.model_names[0]][0]
            ][1:3]
        else:
            print("\033[31m" + "Your device didn't support." + "\033[0m")
            exit()

    def _rdkx5_call(self, input_tensors):
        if not isinstance(input_tensors, (list, tuple)):
            raise TypeError(
                f"Expected input_tensors to be a list or tuple, got {type(input_tensors).__name__}"
            )
        if len(input_tensors) != 1:
            raise ValueError(
                f"device {self.device} input tensor num must be 1, got {len(input_tensors)}."
            )
        for i, tensor in enumerate(input_tensors):
            if not isinstance(tensor, np.ndarray):
                raise TypeError(
                    f"Element {i} of input_tensors is not a numpy.ndarray, got {type(tensor).__name__}"
                )
        begin_time = time()
        quantize_outputs = self.m[0].forward(input_tensors[0])
        logger.debug(
            "\033[1;31m"
            + f"forward time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in quantize_outputs]
        logger.debug(
            "\033[1;31m"
            + f"c to numpy time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        return outputs

    def _rdks100_call(self, input_tensors):
        if not isinstance(input_tensors, (list, tuple)):
            raise TypeError(
                f"Expected input_tensors to be a list or tuple, got {type(input_tensors).__name__}"
            )
        if len(input_tensors) == 0:
            raise ValueError(
                f"device {self.device} input tensor num must >= 1, got {len(input_tensors)}."
            )
        for i, tensor in enumerate(input_tensors):
            if not isinstance(tensor, np.ndarray):
                raise TypeError(
                    f"Element {i} of input_tensors is not a numpy.ndarray, got {type(tensor).__name__}"
                )
        begin_time = time()
        outputs = [
            self.m.run(
                {
                    self.m.input_names[self.m.model_names[0]][idx]: input_tensor
                    for idx, input_tensor in enumerate(input_tensors)
                }
            )[self.m.model_names[0]][name]
            for name in self.m.output_names[self.m.model_names[0]]
        ]
        logger.debug(
            "\033[1;31m"
            + f"forward time = {1000*(time() - begin_time):.2f} ms"
            + "\033[0m"
        )
        return outputs

    def _rdkx5_str(self):
        info = f"device name: {self.device}"
        info += f"input (H, W): ({self.input_H}, {self.input_W}) \n"
        info += f"\033[1;32m" + "-> input tensors" + "\033[0m \n"
        for i, quantize_input in enumerate(self.m[0].inputs):
            info += f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape} \n"
        info += f"\033[1;32m" + "-> output tensors" + "\033[0m \n"
        for i, quantize_input in enumerate(self.m[0].outputs):
            info += f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape} \n"
        return info

    def _rdks100_str(self):
        info = f"device name: {self.device}"
        info += f"input (H, W): ({self.input_H}, {self.input_W}) \n"
        info += f"model_name: {self.m.model_names[0]} \n"
        info += "\033[1;32m" + "-> input tensors" + "\033[0m \n"
        for i, input_name in enumerate(self.m.input_names[self.m.model_names[0]]):
            info += f"intput[{i}], name={input_name}, type={self.m.input_dtypes[self.m.model_names[0]][input_name]}, shape={self.m.input_shapes[self.m.model_names[0]][input_name]} \n"
        info += "\033[1;32m" + "-> output tensors" + "\033[0m \n"
        for i, output_name in enumerate(self.m.output_names[self.m.model_names[0]]):
            info += f"intput[{i}], name={output_name}, type={self.m.output_dtypes[self.m.model_names[0]][output_name]}, shape={self.m.output_shapes[self.m.model_names[0]][output_name]} \n"
        return info

    def __call__(self, input_tensors):
        if self.device == "rdkx5":
            return self._rdkx5_call(input_tensors)
        if self.device == "rdks100":
            return self._rdks100_call(input_tensors)

    def __str__(self):
        if self.device == "rdkx5":
            return self._rdkx5_str()
        if self.device == "rdks100":
            return self._rdks100_str()

    def __repr__(self):
        return self.__str__()
