/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2025, D-Robotics.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * @file main.cc
 * @brief YOLO26 Object Detection inference on RDK S100 (Nash-E)
 */

#define LETTERBOX_TYPE 1
#define RESIZE_TYPE 0
#define PREPROCESS_TYPE LETTERBOX_TYPE

#define CLASSES_NUM 80
#define NMS_THRESHOLD 0.45f
#define SCORE_THRESHOLD 0.25f

#define FONT_SCALE 0.6
#define FONT_THICKNESS 2
#define BOX_THICKNESS 2

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#include "../common/common.h"

int main(int argc, char** argv) {
    std::string project_root = getProjectRoot();
    
    std::string model_path = "yolo26n_detect.hbm";
    std::string img_path = project_root + "/resource/assets/bus.jpg";
    std::string save_path = "yolo26_detect_result.jpg";
    std::string class_names_path = getDefaultLabelPath(project_root, "detect");

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) img_path = argv[2];
    if (argc >= 4) save_path = argv[3];
    if (argc >= 5) class_names_path = argv[4];

    std::vector<std::string> class_names = loadClassNames(class_names_path);
    if (class_names.empty()) {
        LOG_ERROR("Failed to load class names from: " << class_names_path);
        return -1;
    }

    auto start_load = std::chrono::high_resolution_clock::now();

    hbDNNPackedHandle_t packed_handle;
    const char* model_fn = model_path.c_str();
    CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_handle, &model_fn, 1), "Model init failed");
    
    const char** model_names;
    int model_count = 0;
    CHECK_SUCCESS(hbDNNGetModelNameList(&model_names, &model_count, packed_handle), "Get model name failed");
    
    hbDNNHandle_t dnn_handle;
    CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_handle, model_names[0]), "Get handle failed");

    auto dur_load = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_load).count() / 1000.0;

    hbDNNTensorProperties in_props;
    CHECK_SUCCESS(hbDNNGetInputTensorProperties(&in_props, dnn_handle, 0), "Get input props failed");
    
    int input_h = in_props.validShape.dimensionSize[1];
    int input_w = in_props.validShape.dimensionSize[2];
    
    if (input_h <= 3 && in_props.validShape.dimensionSize[2] > 3) {
        input_h = in_props.validShape.dimensionSize[2];
        input_w = in_props.validShape.dimensionSize[3];
    }

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) { 
        LOG_ERROR("Failed to load image: " << img_path);
        return -1; 
    }
    
    auto start_pre = std::chrono::high_resolution_clock::now();
    
    float x_scale, y_scale;
    int x_shift, y_shift;
    bool use_letterbox = (PREPROCESS_TYPE == LETTERBOX_TYPE);
    cv::Mat preprocessed_img = preprocessImage(img, input_h, input_w, 
                                                     x_scale, y_scale, x_shift, y_shift, 
                                                     use_letterbox);
    cv::Mat nv12_img_full = bgr2nv12(preprocessed_img);

    auto dur_pre = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_pre).count() / 1000.0;

    int32_t input_count = 0;
    CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "Get input count failed");

    std::vector<hbDNNTensor> input_tensors(input_count);

    for (int i = 0; i < input_count; ++i) {
        CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i), 
                      "Get props failed");
        int data_size = 0;
        
        if (i == 0) {
            data_size = input_h * input_w;
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = input_h;
            input_tensors[i].properties.validShape.dimensionSize[2] = input_w;
            input_tensors[i].properties.validShape.dimensionSize[3] = 1;
            input_tensors[i].properties.stride[3] = 1;
            input_tensors[i].properties.stride[2] = 1;
            input_tensors[i].properties.stride[1] = input_w;
            input_tensors[i].properties.stride[0] = input_h * input_w;
        } else {
            int uv_h = input_h / 2;
            int uv_w = input_w / 2;
            data_size = uv_h * uv_w * 2;
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = uv_h;
            input_tensors[i].properties.validShape.dimensionSize[2] = uv_w;
            input_tensors[i].properties.validShape.dimensionSize[3] = 2;
            input_tensors[i].properties.stride[3] = 1;
            input_tensors[i].properties.stride[2] = 2;
            input_tensors[i].properties.stride[1] = uv_w * 2;
            input_tensors[i].properties.stride[0] = uv_h * uv_w * 2;
        }
        
        CHECK_SUCCESS(hbUCPMallocCached(&input_tensors[i].sysMem, data_size, 0), "Malloc failed");
        
        if (i == 0) {
            memcpy(input_tensors[i].sysMem.virAddr, nv12_img_full.data, data_size);
        } else {
            memcpy(input_tensors[i].sysMem.virAddr, 
                   nv12_img_full.data + input_h * input_w, data_size);
        }
        
        hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    int output_cnt = 0;
    hbDNNGetOutputCount(&output_cnt, dnn_handle);
    
    std::vector<hbDNNTensor> outputs(output_cnt);
    for (int i = 0; i < output_cnt; ++i) {
        hbDNNGetOutputTensorProperties(&outputs[i].properties, dnn_handle, i);
        hbUCPMallocCached(&outputs[i].sysMem, outputs[i].properties.alignedByteSize, 0);
    }

    auto start_infer = std::chrono::high_resolution_clock::now();
    
    hbUCPTaskHandle_t task = nullptr;
    CHECK_SUCCESS(hbDNNInferV2(&task, outputs.data(), input_tensors.data(), dnn_handle), "Infer failed");
    
    hbUCPSchedParam param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&param);
    param.backend = HB_UCP_BPU_CORE_ANY;
    
    CHECK_SUCCESS(hbUCPSubmitTask(task, &param), "Submit failed");
    CHECK_SUCCESS(hbUCPWaitTaskDone(task, 0), "Wait failed");
    
    auto dur_infer = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_infer).count() / 1000.0;

    auto start_post = std::chrono::high_resolution_clock::now();
    
    int H_8 = input_h / 8;
    int H_16 = input_h / 16;
    int H_32 = input_h / 32;
    
    int W_8 = input_w / 8;
    int W_16 = input_w / 16;
    int W_32 = input_w / 32;
    
    // Output pairs: [Box_8, Cls_8, Box_16, Cls_16, Box_32, Cls_32]
    int order[6] = {0};
    
    for (int i = 0; i < output_cnt; ++i) {
        hbUCPMemFlush(&outputs[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
        
        int h = outputs[i].properties.validShape.dimensionSize[1];
        int w = outputs[i].properties.validShape.dimensionSize[2];
        int c = outputs[i].properties.validShape.dimensionSize[3];
        
        if (h == H_8 && w == W_8) {
            if (c == 4) order[0] = i;
            else order[1] = i;
        } else if (h == H_16 && w == W_16) {
            if (c == 4) order[2] = i;
            else order[3] = i;
        } else if (h == H_32 && w == W_32) {
            if (c == 4) order[4] = i;
            else order[5] = i;
        }
    }

    std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM);
    std::vector<std::vector<float>> scores(CLASSES_NUM);
    float conf_thres_raw = calcLogitThreshold(SCORE_THRESHOLD);

    int strides[] = {8, 16, 32};
    for (int s = 0; s < 3; ++s) {
        int box_idx = order[s * 2];
        int cls_idx = order[s * 2 + 1];
        int stride = strides[s];
        int h = input_h / stride;
        int w = input_w / stride;
        
        float* box_ptr = static_cast<float*>(outputs[box_idx].sysMem.virAddr);
        float* cls_ptr = static_cast<float*>(outputs[cls_idx].sysMem.virAddr);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int offset = i * w + j;
                float* cur_cls = cls_ptr + offset * CLASSES_NUM;
                float* cur_box = box_ptr + offset * 4;

                int max_id = 0;
                for (int k = 1; k < CLASSES_NUM; ++k) {
                    if (cur_cls[k] > cur_cls[max_id]) max_id = k;
                }
                
                if (cur_cls[max_id] < conf_thres_raw) continue;

                // Box format: [l, t, r, b]
                // Grid center: (j + 0.5, i + 0.5)
                float l = cur_box[0];
                float t = cur_box[1];
                float r = cur_box[2];
                float b = cur_box[3];
                
                float grid_center_x = j + 0.5f;
                float grid_center_y = i + 0.5f;
                
                float x1 = (grid_center_x - l) * stride;
                float y1 = (grid_center_y - t) * stride;
                float x2 = (grid_center_x + r) * stride;
                float y2 = (grid_center_y + b) * stride;
                
                float score = sigmoid(cur_cls[max_id]);
                
                bboxes[max_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[max_id].push_back(score);
            }
        }
    }

    std::vector<Detection> dets;
    for (int c = 0; c < CLASSES_NUM; ++c) {
        if (bboxes[c].empty()) continue;
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes[c], scores[c], SCORE_THRESHOLD, NMS_THRESHOLD, indices);
        for (int idx : indices) {
            dets.emplace_back(c, scores[c][idx], bboxes[c][idx]);
        }
    }
    
    auto dur_post = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_post).count() / 1000.0;

    std::cout << "\033[1;31m[Detect] Load Model time = " << std::fixed << std::setprecision(2) << dur_load << " ms\033[0m" << std::endl;
    std::cout << "\033[1;31m[Detect] Pre Process time = " << std::fixed << std::setprecision(2) << dur_pre << " ms\033[0m" << std::endl;
    std::cout << "\033[1;31m[Detect] Forward time = " << std::fixed << std::setprecision(2) << dur_infer << " ms\033[0m" << std::endl;
    std::cout << "\033[1;31m[Detect] Post Process time = " << std::fixed << std::setprecision(2) << dur_post << " ms\033[0m" << std::endl;

    printDetections(dets, class_names, x_scale, y_scale, x_shift, y_shift);

    drawDetections(img, dets, class_names, x_scale, y_scale, x_shift, y_shift,
                        FONT_SCALE, FONT_THICKNESS, BOX_THICKNESS);
    
    cv::imwrite(save_path, img);
    std::cout << "\n[Saved] Result saved to: " << save_path << std::endl;

    hbUCPReleaseTask(task);
    for (auto& t : input_tensors) hbUCPFree(&t.sysMem);
    for (auto& t : outputs) hbUCPFree(&t.sysMem);
    hbDNNRelease(packed_handle);
    
    return 0;
}
