"""
 Copyright (c) 2021-2024 D-Robotics Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import time
import os
import sys

import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def export_onnx():
    args = parse_args()
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    print('Loading model from %s' % model_path)
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    print(f"Input shape: {dummy_input.shape}")
    with torch.no_grad():
        outputs = model(dummy_input)
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                print(f"Output '{k}' shape: {v.shape}")
        else:
            print(f"Output shape: {outputs.shape}")
    onnx_path = os.path.splitext(model_path)[0] + ".onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["instance_seg_logits", "binary_seg_pred","binary_seg_logits"],
        opset_version=14,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Exported ONNX model to {onnx_path}")

def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img

def test():
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args()
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    print('Loading model from %s' % model_path)
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    input = Image.open(img_path)
    input = input.resize((resize_width, resize_height))
    input = np.array(input)
    print(outputs.keys())
    instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255

    cv2.imwrite(os.path.join('test_output', 'input.jpg'), input)
    cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'), instance_pred.transpose((1, 2, 0)))
    print(f"Instance prediction shape: {instance_pred.shape}")
    cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), binary_pred)
    print(f"Instance prediction shape: {binary_pred.shape}")

if __name__ == "__main__":
    test()
    export_onnx()