#!/usr/bin/env python3

from hobot_dnn import pyeasy_dnn as dnn
import numpy as np
import cv2
import time
import ctypes
import json

from hobot_structures import (
    hbDNNTensor_t,
    ClassificationPostProcessInfo_t
)

from utils import bgr2nv12_opencv, print_properties, get_hw

output_tensors = None

libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so')

get_Postprocess_result = libpostprocess.ClassificationPostProcess
get_Postprocess_result.argtypes = [
    ctypes.POINTER(ClassificationPostProcessInfo_t)]
get_Postprocess_result.restype = ctypes.c_char_p

if __name__ == '__main__':
    
    models = dnn.load('./model/efficientnet_lite3_300x300_nv12.hbm')
    img_file = cv2.imread('./data/Scottish_deerhound.JPEG')
    
    print("=" * 10, "inputs[0] properties", "=" * 10)
    print_properties(models[0].inputs[0].properties)
    print("inputs[0] name is:", models[0].inputs[0].name)

    print("=" * 10, "outputs[0] properties", "=" * 10)
    print_properties(models[0].outputs[0].properties)
    print("outputs[0] name is:", models[0].outputs[0].name)

    h, w = get_hw(models[0].inputs[0].properties)
    des_dim = (w, h)
    resized_data = cv2.resize(img_file, des_dim, interpolation=cv2.INTER_AREA)
    nv12_data = bgr2nv12_opencv(resized_data)

    outputs = models[0].forward(nv12_data)
    
    t0 = time.time()
    # 获取结构体信息
    classification_postprocess_info = ClassificationPostProcessInfo_t()
    classification_postprocess_info.height = h
    classification_postprocess_info.width = w
    org_height, org_width = img_file.shape[0:2]
    classification_postprocess_info.ori_height = org_height
    classification_postprocess_info.ori_width = org_width
    classification_postprocess_info.score_threshold = 0.3
    classification_postprocess_info.nms_threshold = 0
    classification_postprocess_info.nms_top_k = 5
    classification_postprocess_info.is_pad_resize = 0
    classification_postprocess_info.use_softmax = True

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    
    for i in range(len(models[0].outputs)):
        if (len(outputs[i].properties.scale_data) == 0):
            output_tensors[i].properties.quantiType = 0
            output_tensors[i].sysMem.virAddr = ctypes.cast(
                outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
        else:
            output_tensors[i].properties.quantiType = 1
            output_tensors[i].properties.scale.scaleData = outputs[i].properties.scale_data.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float))
            output_tensors[i].sysMem.virAddr = ctypes.cast(
                outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)

        for j in range(len(outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.numDimensions = len(
                outputs[i].properties.shape)
            output_tensors[i].properties.validShape.dimensionSize[j] = outputs[i].properties.shape[j]

        libpostprocess.ClassificationDoProcess(
            output_tensors[i], ctypes.pointer(classification_postprocess_info), i)

    result_str = get_Postprocess_result(
        ctypes.pointer(classification_postprocess_info))
    result_str = result_str.decode('utf-8')
    
    t1 = time.time()
    print("postprocess time is :", (t1 - t0))

    data = json.loads(result_str[25:])

    # 遍历每一个结果
    for result in data:
        prob = result['prob']  # 得分
        label = result['label']  # id
        name = result['class_name']  # 类别名称

        # 打印信息
        print(f"cls id: {label}, Confidence: {prob}, class_name: {name}")
