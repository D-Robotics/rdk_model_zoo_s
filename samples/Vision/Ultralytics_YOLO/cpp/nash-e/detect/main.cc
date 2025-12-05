/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

Copyright (c) 2024-2025, D-Robotics.

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

// 注意: 此程序在RDK S100 (Nash-E) 板端运行
// Attention: This program runs on RDK S100 (Nash-E) board.

// ============================================================================ 
// Configuration Parameters
// ============================================================================ 

#define MODEL_PATH "yolo11n_detect_nashe_640x640_nv12.hbm"
#define TEST_IMG_PATH "../../../../../../resource/datasets/COCO2017/assets/bus.jpg"
#define IMG_SAVE_PATH "detect_result_nashe.jpg"

// 前处理方式: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 模型参数 / Model Parameters
#define CLASSES_NUM 80
#define NMS_THRESHOLD 0.45
#define SCORE_THRESHOLD 0.25
#define NMS_TOP_K 300
#define REG 16

// 可视化参数 / Visualization Parameters
#define FONT_SCALE 0.6
#define FONT_THICKNESS 2
#define BOX_THICKNESS 2

// ============================================================================ 
// Includes
// ============================================================================ 

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// RDK S100 UCP API
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

// ============================================================================ 
// Macros & Helpers
// ============================================================================ 

#define CHECK_SUCCESS(value, errmsg)                                         \
    do {                                                                     \
        auto ret_code = value;                                               \
        if (ret_code != 0) {                                                 \
            std::cerr << "\033[1;31m[ERROR]\033[0m " << __FILE__ << ":"     \
                      << __LINE__ << " " << errmsg                           \
                      << ", error code: " << ret_code << std::endl;          \
            return ret_code;                                                 \
        }                                                                    \
    } while (0)

#define LOG_INFO(msg) \
    std::cout << "\033[1;32m[INFO]\033[0m " << msg << std::endl

#define LOG_WARN(msg) \
    std::cout << "\033[1;33m[WARN]\033[0m " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl

#define LOG_TIME(msg, duration) \
    std::cout << "\033[1;31m" << msg << " = " << std::fixed            \
              << std::setprecision(2) << (duration) << " ms\033[0m"    \
              << std::endl

const std::vector<std::string> COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(56, 56, 255),    cv::Scalar(151, 157, 255),
    cv::Scalar(31, 112, 255),   cv::Scalar(29, 178, 255),
    cv::Scalar(49, 210, 207),   cv::Scalar(10, 249, 72),
    cv::Scalar(23, 204, 146),   cv::Scalar(134, 219, 61),
    cv::Scalar(52, 147, 26),    cv::Scalar(187, 212, 0),
    cv::Scalar(168, 153, 44),   cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52),    cv::Scalar(255, 115, 100),
    cv::Scalar(236, 24, 0),     cv::Scalar(255, 56, 132),
    cv::Scalar(133, 0, 82),     cv::Scalar(255, 56, 203),
    cv::Scalar(200, 149, 255),  cv::Scalar(199, 55, 255)
};

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2d bbox;
    Detection(int id, float conf, const cv::Rect2d& box) : class_id(id), confidence(conf), bbox(box) {}
};

// ============================================================================ 
// Utility Functions
// ============================================================================ 

// Softmax function
void softmax(float* input, float* output, int size) {
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

cv::Mat bgr2nv12(const cv::Mat& bgr_img) {
    auto start = std::chrono::high_resolution_clock::now();

    int h = bgr_img.rows, w = bgr_img.cols;
    cv::Mat yuv;
    cv::cvtColor(bgr_img, yuv, cv::COLOR_BGR2YUV_I420);
    cv::Mat nv12(h * 3 / 2, w, CV_8UC1);
    uint8_t* y_ptr = nv12.ptr<uint8_t>();
    uint8_t* uv_ptr = y_ptr + h * w;
    uint8_t* u_src = yuv.ptr<uint8_t>() + h * w;
    uint8_t* v_src = u_src + (h/2) * (w/2);
    
    // Copy Y
    memcpy(y_ptr, yuv.ptr<uint8_t>(), h * w);
    
    // Interleave U and V
    for (int i = 0; i < (h/2) * (w/2); ++i) {
        *uv_ptr++ = *u_src++;
        *uv_ptr++ = *v_src++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("BGR to NV12 time", duration);

    return nv12;
}

cv::Mat preprocess_image(const cv::Mat& img, int input_h, int input_w, float& x_scale, float& y_scale, int& x_shift, int& y_shift) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;
    
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) { // LetterBox
        x_scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        y_scale = x_scale;
        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);
        
        x_shift = (input_w - new_w) / 2;
        y_shift = (input_h - new_h) / 2;
        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, y_shift, input_h - new_h - y_shift, x_shift, input_w - new_w - x_shift, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (LetterBox) time", duration);
        
    } else { // Resize
        cv::resize(img, result, cv::Size(input_w, input_h));
        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
        x_shift = 0; y_shift = 0;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (Resize) time", duration);
    }
    
    LOG_INFO("Scale: x=" << x_scale << ", y=" << y_scale);
    LOG_INFO("Shift: x=" << x_shift << ", y=" << y_shift);
    return result;
}

void draw_detections(cv::Mat& img, const std::vector<Detection>& detections,
                    float x_scale, float y_scale, int x_shift, int y_shift) {
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& det : detections) {
        // Transform coordinates back to original image space
        float x1 = (det.bbox.x - x_shift) / x_scale;
        float y1 = (det.bbox.y - y_shift) / y_scale;
        float x2 = x1 + det.bbox.width / x_scale;
        float y2 = y1 + det.bbox.height / y_scale;

        // Clamp coordinates
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img.cols)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img.rows)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img.cols)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img.rows)));

        // Get color
        cv::Scalar color = COLORS[det.class_id % COLORS.size()];

        // Draw box
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                     color, BOX_THICKNESS);

        // Prepare label
        std::string label = COCO_NAMES[det.class_id] + ": " +
                           std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        // Get text size
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             FONT_SCALE, FONT_THICKNESS, &baseline);

        // Draw label background
        int label_y = std::max(static_cast<int>(y1), text_size.height + 10);
        cv::rectangle(img,
                     cv::Point(x1, label_y - text_size.height - 10),
                     cv::Point(x1 + text_size.width, label_y),
                     color, cv::FILLED);

        // Draw label text
        cv::putText(img, label, cv::Point(x1, label_y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                   cv::Scalar(255, 255, 255), FONT_THICKNESS, cv::LINE_AA);

        // Print detection info
        std::cout << "  (" << static_cast<int>(x1) << ", " << static_cast<int>(y1)
                  << ", " << static_cast<int>(x2) << ", " << static_cast<int>(y2)
                  << ") -> " << label << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("Draw results time", duration);
}

// ============================================================================ 
// Main
// ============================================================================ 

int main(int argc, char** argv) {
    LOG_INFO("=== Ultralytics YOLO Detect Demo (S100 Nash-E UCP) ===");
    LOG_INFO("OpenCV Version: " << CV_VERSION);
    
    // 1. Parse Args
    std::string model_path = MODEL_PATH;
    std::string img_path = TEST_IMG_PATH;
    std::string save_path = IMG_SAVE_PATH;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) img_path = argv[2];
    if (argc >= 4) save_path = argv[3];

    // 2. Init Model
    LOG_INFO("Loading model: " << model_path);
    auto start_load = std::chrono::high_resolution_clock::now();

    hbDNNPackedHandle_t packed_handle;
    const char* model_fn = model_path.c_str();
    CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_handle, &model_fn, 1), "Init failed");
    
    const char** model_names;
    int model_count = 0;
    CHECK_SUCCESS(hbDNNGetModelNameList(&model_names, &model_count, packed_handle), "Get name failed");
    if (model_count > 1) {
        LOG_WARN("Model file contains " << model_count << " models, using the first one");
    }
    
    hbDNNHandle_t dnn_handle;
    CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_handle, model_names[0]), "Get handle failed");

    auto dur_load = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_load).count() / 1000.0;
    LOG_TIME("Load model time", dur_load);

    // 3. Prepare Input Properties & Check Info
    hbDNNTensorProperties in_props;
    CHECK_SUCCESS(hbDNNGetInputTensorProperties(&in_props, dnn_handle, 0), "Get input props failed");
    
    int input_h = in_props.validShape.dimensionSize[1];
    int input_w = in_props.validShape.dimensionSize[2];
    
    // Fallback logic for dimension if layout is strange or different on S100
    if (input_h <= 3 && in_props.validShape.dimensionSize[2] > 3) {
        input_h = in_props.validShape.dimensionSize[2];
        input_w = in_props.validShape.dimensionSize[3];
    }
    LOG_INFO("Model Input Shape: " << input_w << "x" << input_h);

    // 4. Load Image
    LOG_INFO("Loading image: " << img_path);
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) { 
        LOG_ERROR("Failed to load image from path: " << img_path);
        return -1; 
    }
    
    float x_scale, y_scale;
    int x_shift, y_shift;
    cv::Mat preprocessed_img = preprocess_image(img, input_h, input_w, x_scale, y_scale, x_shift, y_shift);
    cv::Mat nv12_img_full = bgr2nv12(preprocessed_img);

    // 5. Prepare Input Tensors (Multi-Input: Y and UV split for S100)
    int32_t input_count = 0;
    CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "Failed to get input count");
    LOG_INFO("Model Input Count: " << input_count);

    std::vector<hbDNNTensor> input_tensors(input_count);

    for (int i = 0; i < input_count; ++i) {
        CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i), "Get props failed");
        int data_size = 0;
        // S100 specific: Split NV12 into Y and UV planes if input_count > 1
        // If input_count == 1, this loop runs once and logic should be adjusted, 
        // but here we stick to the reference Nash-E implementation logic.
        if (i == 0) { // Y Plane
            data_size = input_h * input_w;
            // Manually set properties if needed, or rely on GetInputTensorProperties? 
            // Reference code manually sets them, implying GetProperties might return placeholder dims for split inputs.
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = input_h;
            input_tensors[i].properties.validShape.dimensionSize[2] = input_w;
            input_tensors[i].properties.validShape.dimensionSize[3] = 1;
            input_tensors[i].properties.stride[3] = 1;
            input_tensors[i].properties.stride[2] = 1;
            input_tensors[i].properties.stride[1] = input_w;
            input_tensors[i].properties.stride[0] = input_h * input_w;
        } else { // UV Plane
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
        
        // UCP Malloc
        CHECK_SUCCESS(hbUCPMallocCached(&input_tensors[i].sysMem, data_size, 0), "Malloc failed");
        
        // Copy Data
        if (i == 0) memcpy(input_tensors[i].sysMem.virAddr, nv12_img_full.data, data_size);
        else memcpy(input_tensors[i].sysMem.virAddr, nv12_img_full.data + input_h * input_w, data_size);
        
        // Flush
        hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    // 6. Prepare Output Tensors
    int output_cnt = 0;
    hbDNNGetOutputCount(&output_cnt, dnn_handle);
    LOG_INFO("Model Output Count: " << output_cnt);
    
    std::vector<hbDNNTensor> outputs(output_cnt);
    for (int i = 0; i < output_cnt; ++i) {
        hbDNNGetOutputTensorProperties(&outputs[i].properties, dnn_handle, i);
        hbUCPMallocCached(&outputs[i].sysMem, outputs[i].properties.alignedByteSize, 0);
    }

    // 7. Infer
    LOG_INFO("Running inference...");
    auto start_infer = std::chrono::high_resolution_clock::now();
    
    hbUCPTaskHandle_t task = nullptr;
    CHECK_SUCCESS(hbDNNInferV2(&task, outputs.data(), input_tensors.data(), dnn_handle), "Infer failed");
    
    hbUCPSchedParam param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&param);
    param.backend = HB_UCP_BPU_CORE_ANY;
    
    CHECK_SUCCESS(hbUCPSubmitTask(task, &param), "Submit failed");
    CHECK_SUCCESS(hbUCPWaitTaskDone(task, 0), "Wait failed");
    
    auto dur_infer = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_infer).count() / 1000.0;
    LOG_TIME("Inference time", dur_infer);

    // 8. Post Process
    LOG_INFO("Post-processing...");
    auto start_post = std::chrono::high_resolution_clock::now();
    
    // Identify Output Order
    int H_8 = input_h/8;
    int H_16 = input_h/16;
    int H_32 = input_h/32;
    
    int W_8 = input_w/8;
    int W_16 = input_w/16;
    int W_32 = input_w/32;
    
    int order[6] = {0};
    
    // Mapping logic: Find indices for 8, 16, 32 strides
    // Expected: [cls8, box8, cls16, box16, cls32, box32]
    for(int i=0; i<output_cnt; ++i) {
        hbUCPMemFlush(&outputs[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
        
        int h = outputs[i].properties.validShape.dimensionSize[1];
        int w = outputs[i].properties.validShape.dimensionSize[2];
        int c = outputs[i].properties.validShape.dimensionSize[3];
        
        // Robust mapping
        if (h == H_8 && w == W_8) {
            if (c == CLASSES_NUM) order[0] = i; // cls 8
            else order[1] = i;                  // box 8
        } else if (h == H_16 && w == W_16) {
            if (c == CLASSES_NUM) order[2] = i; // cls 16
            else order[3] = i;                  // box 16
        } else if (h == H_32 && w == W_32) {
            if (c == CLASSES_NUM) order[4] = i; // cls 32
            else order[5] = i;                  // box 32
        }
    }
    LOG_INFO("Output Mapping: " << order[0] << "," << order[1] << " | " 
             << order[2] << "," << order[3] << " | " << order[4] << "," << order[5]);

    std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM);
    std::vector<std::vector<float>> scores(CLASSES_NUM);
    float conf_thres_raw = -std::log(1.0f / SCORE_THRESHOLD - 1.0f);

    int strides[] = {8, 16, 32};
    for (int s = 0; s < 3; ++s) {
        int cls_idx = order[s*2];
        int box_idx = order[s*2+1];
        int stride = strides[s];
        int h = input_h / stride, w = input_w / stride;
        
        float* cls_ptr = (float*)outputs[cls_idx].sysMem.virAddr;
        float* box_ptr = (float*)outputs[box_idx].sysMem.virAddr;

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int offset = i * w + j;
                float* cur_cls = cls_ptr + offset * CLASSES_NUM;
                float* cur_box = box_ptr + offset * 4 * REG;

                int max_id = 0;
                for (int k = 1; k < CLASSES_NUM; ++k) {
                    if (cur_cls[k] > cur_cls[max_id]) max_id = k;
                }
                
                if (cur_cls[max_id] < conf_thres_raw) continue;

                // Box decoding
                float ltrb[4] = {0};
                for (int k = 0; k < 4; ++k) {
                    float dfl_values[REG];
                    float softmax_values[REG];
                    
                    for (int r = 0; r < REG; ++r) {
                        dfl_values[r] = cur_box[k*REG + r];
                    }
                    softmax(dfl_values, softmax_values, REG);
                    for (int r = 0; r < REG; ++r) ltrb[k] += softmax_values[r] * r;
                }

                float x1 = (j + 0.5f - ltrb[0]) * stride;
                float y1 = (i + 0.5f - ltrb[1]) * stride;
                float x2 = (j + 0.5f + ltrb[2]) * stride;
                float y2 = (i + 0.5f + ltrb[3]) * stride;
                
                bboxes[max_id].push_back(cv::Rect2d(x1, y1, x2-x1, y2-y1));
                scores[max_id].push_back(1.0f / (1.0f + std::exp(-cur_cls[max_id])));
            }
        }
    }

    // NMS
    std::vector<Detection> dets;
    for (int c = 0; c < CLASSES_NUM; ++c) {
        if (bboxes[c].empty()) continue;
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes[c], scores[c], SCORE_THRESHOLD, NMS_THRESHOLD, indices);
        for (int idx : indices) dets.emplace_back(c, scores[c][idx], bboxes[c][idx]);
    }
    
    auto dur_post = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_post).count() / 1000.0;
    LOG_TIME("Post-processing time", dur_post);
    LOG_INFO("Detected: " << dets.size() << " objects");

    // 9. Draw and Save
    if (!dets.empty()) {
        draw_detections(img, dets, x_scale, y_scale, x_shift, y_shift);
    }
    
    cv::imwrite(save_path, img);
    LOG_INFO("Result saved to: " << save_path);

    // 10. Cleanup
    hbUCPReleaseTask(task);
    for(auto& t : input_tensors) hbUCPFree(&t.sysMem);
    for(auto& t : outputs) hbUCPFree(&t.sysMem);
    hbDNNRelease(packed_handle);
    
    LOG_INFO("=== Demo completed successfully ===");
    return 0;
}
