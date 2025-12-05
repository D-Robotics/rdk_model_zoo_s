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

#define MODEL_PATH "yolo11n_pose_nashe_640x640_nv12.hbm"
#define TEST_IMG_PATH "../../../../../../resource/datasets/COCO2017/assets/bus.jpg"
#define IMG_SAVE_PATH "pose_result_nashe.jpg"

// 前处理方式: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

#define CLASSES_NUM 1
#define NMS_THRESHOLD 0.45
#define SCORE_THRESHOLD 0.25
#define KPT_SCORE_THRESHOLD 0.5
#define REG 16
#define KPT_NUM 17
#define NMS_TOP_K 300

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

const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
    cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
    cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
    cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
    cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)
};

struct PoseDetection {
    cv::Rect2d bbox;
    float score;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> kpt_scores;
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

cv::Mat preprocess_image(const cv::Mat& img, int input_h, int input_w, float& x_scale, float& y_scale, int& x_shift, int& y_shift) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;
    
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        x_scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        y_scale = x_scale;
        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);
        
        x_shift = (input_w - new_w) / 2;
        y_shift = (input_h - new_h) / 2;
        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, y_shift, input_h - new_h - y_shift, x_shift, input_w - new_w - x_shift, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else {
        cv::resize(img, result, cv::Size(input_w, input_h));
        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
        x_shift = 0; y_shift = 0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("Preprocess time", duration);
    return result;
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

void draw_pose(cv::Mat& img, const std::vector<PoseDetection>& detections, 
               float x_scale, float y_scale, int x_shift, int y_shift) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for(const auto& det : detections) {
        float x1 = (det.bbox.x - x_shift) / x_scale;
        float y1 = (det.bbox.y - y_shift) / y_scale;
        float w = det.bbox.width / x_scale;
        float h = det.bbox.height / y_scale;
        
        // Draw Box
        cv::rectangle(img, cv::Rect(x1, y1, w, h), cv::Scalar(0,255,0), 2);
        
        // Draw Skeleton
        for(const auto& edge : SKELETON) {
            if (det.kpt_scores[edge.first] < KPT_SCORE_THRESHOLD || det.kpt_scores[edge.second] < KPT_SCORE_THRESHOLD) continue;
            
            cv::Point p1((det.keypoints[edge.first].x - x_shift) / x_scale, (det.keypoints[edge.first].y - y_shift) / y_scale);
            cv::Point p2((det.keypoints[edge.second].x - x_shift) / x_scale, (det.keypoints[edge.second].y - y_shift) / y_scale);
            cv::line(img, p1, p2, COLORS[edge.first % COLORS.size()], 2);
        }
        
        // Draw Keypoints
        for(int k=0; k<KPT_NUM; ++k) {
            if(det.kpt_scores[k] < KPT_SCORE_THRESHOLD) continue;
            cv::Point p((det.keypoints[k].x - x_shift) / x_scale, (det.keypoints[k].y - y_shift) / y_scale);
            cv::circle(img, p, 4, COLORS[k % COLORS.size()], -1);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("Draw time", duration);
}

// ============================================================================ 
// Main
// ============================================================================ 

int main(int argc, char** argv) {
    LOG_INFO("=== Ultralytics YOLO Pose Demo (S100 Nash-E UCP) ===");
    LOG_INFO("OpenCV Version: " << CV_VERSION);
    
    std::string model_path = MODEL_PATH;
    std::string img_path = TEST_IMG_PATH;
    std::string save_path = IMG_SAVE_PATH;
    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) img_path = argv[2];
    if (argc >= 4) save_path = argv[3];

    // 1. Init Model
    LOG_INFO("Loading model: " << model_path);
    auto start_load = std::chrono::high_resolution_clock::now();
    
    hbDNNPackedHandle_t packed_handle;
    const char* model_fn = model_path.c_str();
    CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_handle, &model_fn, 1), "Init failed");
    const char** names;
    int count;
    hbDNNGetModelNameList(&names, &count, packed_handle);
    if (count > 1) LOG_WARN("Model file contains " << count << " models, using the first one");
    
    hbDNNHandle_t dnn_handle;
    CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_handle, names[0]), "Get model handle failed");
    
    auto dur_load = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_load).count() / 1000.0;
    LOG_TIME("Load model time", dur_load);

    // 2. Input Properties
    hbDNNTensorProperties in_props;
    CHECK_SUCCESS(hbDNNGetInputTensorProperties(&in_props, dnn_handle, 0), "Get input props failed");
    
    int input_h = in_props.validShape.dimensionSize[1];
    int input_w = in_props.validShape.dimensionSize[2];
    if (input_h <= 3 && in_props.validShape.dimensionSize[2] > 3) {
        input_h = in_props.validShape.dimensionSize[2];
        input_w = in_props.validShape.dimensionSize[3];
    }
    LOG_INFO("Model Input Shape: " << input_w << "x" << input_h);

    // 3. Load Image
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        LOG_ERROR("Failed to load image from path: " << img_path);
        return -1;
    }
    
    float x_scale, y_scale; int x_shift, y_shift;
    cv::Mat preprocessed_img = preprocess_image(img, input_h, input_w, x_scale, y_scale, x_shift, y_shift);
    cv::Mat nv12_img_full = bgr2nv12(preprocessed_img);

    // 4. Prepare Input Tensors
    int32_t input_count = 0;
    CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "Failed to get input count");
    std::vector<hbDNNTensor> input_tensors(input_count);

    for (int i = 0; i < input_count; ++i) {
        CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i), "Get props failed");
        int data_size = 0;
        
        if (input_count > 1) { // S100 Split Mode
            if (i == 0) { // Y
                data_size = input_h * input_w;
                input_tensors[i].properties.validShape.dimensionSize[0] = 1;
                input_tensors[i].properties.validShape.dimensionSize[1] = input_h;
                input_tensors[i].properties.validShape.dimensionSize[2] = input_w;
                input_tensors[i].properties.validShape.dimensionSize[3] = 1;
                input_tensors[i].properties.stride[3] = 1;
                input_tensors[i].properties.stride[2] = 1;
                input_tensors[i].properties.stride[1] = input_w;
                input_tensors[i].properties.stride[0] = input_h * input_w;
            } else { // UV
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
        } else { // Single Input
             data_size = input_h * input_w * 3 / 2;
        }
        
        CHECK_SUCCESS(hbUCPMallocCached(&input_tensors[i].sysMem, data_size, 0), "Malloc failed");
        
        if (input_count > 1) {
            if (i == 0) memcpy(input_tensors[i].sysMem.virAddr, nv12_img_full.data, data_size);
            else memcpy(input_tensors[i].sysMem.virAddr, nv12_img_full.data + input_h * input_w, data_size);
        } else {
            memcpy(input_tensors[i].sysMem.virAddr, nv12_img_full.data, data_size);
        }
        hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    // 5. Prepare Outputs
    int out_cnt;
    hbDNNGetOutputCount(&out_cnt, dnn_handle);
    std::vector<hbDNNTensor> outputs(out_cnt);
    for(int i=0; i<out_cnt; ++i) {
        hbDNNGetOutputTensorProperties(&outputs[i].properties, dnn_handle, i);
        hbUCPMallocCached(&outputs[i].sysMem, outputs[i].properties.alignedByteSize, 0);
    }

    // 6. Infer
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

    // 7. Post Process
    LOG_INFO("Post-processing...");
    auto start_post = std::chrono::high_resolution_clock::now();

    std::vector<PoseDetection> dets;
    int strides[] = {8, 16, 32};
    float conf_thres = -std::log(1.0f/SCORE_THRESHOLD - 1.0f);
    
    int H_8 = input_h / 8, W_8 = input_w / 8;
    int H_16 = input_h / 16, W_16 = input_w / 16;
    int H_32 = input_h / 32, W_32 = input_w / 32;

    // Map outputs: [cls8, box8, kpt8, cls16, box16, kpt16, cls32, box32, kpt32]
    int order[9] = {0};
    
    for(int i=0; i<out_cnt; ++i) {
        hbUCPMemFlush(&outputs[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
        int h = outputs[i].properties.validShape.dimensionSize[1];
        int w = outputs[i].properties.validShape.dimensionSize[2];
        int c = outputs[i].properties.validShape.dimensionSize[3];
        
        int s_idx = -1;
        if (h == H_8 && w == W_8) s_idx = 0;
        else if (h == H_16 && w == W_16) s_idx = 1;
        else if (h == H_32 && w == W_32) s_idx = 2;
        
        if (s_idx != -1) {
            int t_idx = -1;
            // Heuristic for determining output type based on channels
            if (c == CLASSES_NUM) t_idx = 0; // cls
            else if (c == 4 * REG) t_idx = 1; // box (DFL)
            else if (c == KPT_NUM * 3) t_idx = 2; // kpt
            
            if (t_idx != -1) order[s_idx * 3 + t_idx] = i;
        }
    }
    LOG_INFO("Output Mapping: " 
             << order[0] << "," << order[1] << "," << order[2] << " | " 
             << order[3] << "," << order[4] << "," << order[5] << " | " 
             << order[6] << "," << order[7] << "," << order[8]);

    for(int s=0; s<3; ++s) {
        float* cls = (float*)outputs[order[s*3+0]].sysMem.virAddr;
        float* box = (float*)outputs[order[s*3+1]].sysMem.virAddr;
        float* kpt = (float*)outputs[order[s*3+2]].sysMem.virAddr;
        int stride = strides[s];
        int grid_h = input_h/stride, grid_w = input_w/stride;

        for(int h=0; h<grid_h; ++h) {
            for(int w=0; w<grid_w; ++w) {
                int off = h*grid_w + w;
                // Check class score (Single class for pose)
                if(cls[off] < conf_thres) continue;

                // Decode Box (DFL + Softmax)
                float ltrb[4]={0};
                for(int k=0; k<4; ++k) {
                    float dfl[REG];
                    for(int r=0; r<REG; ++r) dfl[r] = box[off*4*REG + k*REG + r];
                    softmax(dfl, dfl, REG); // Reuse softmax function
                    for(int r=0; r<REG; ++r) ltrb[k] += dfl[r]*r;
                }
                
                float cx = (w+0.5f)*stride;
                float cy = (h+0.5f)*stride;
                float x1 = cx - ltrb[0]*stride;
                float y1 = cy - ltrb[1]*stride;
                float x2 = cx + ltrb[2]*stride;
                float y2 = cy + ltrb[3]*stride;
                
                PoseDetection det;
                det.bbox = cv::Rect2d(x1, y1, x2-x1, y2-y1);
                det.score = 1.0f/(1.0f+std::exp(-cls[off]));
                det.keypoints.resize(KPT_NUM);
                det.kpt_scores.resize(KPT_NUM);
                
                // Decode Keypoints
                for(int k=0; k<KPT_NUM; ++k) {
                    float kx = kpt[off*KPT_NUM*3 + k*3 + 0];
                    float ky = kpt[off*KPT_NUM*3 + k*3 + 1];
                    float kc = kpt[off*KPT_NUM*3 + k*3 + 2];
                    
                    // YOLOv8/11 Pose formula: (x*2 + grid - 0.5) * stride
                    det.keypoints[k].x = (kx*2.0f + (w+0.5f) - 0.5f)*stride;
                    det.keypoints[k].y = (ky*2.0f + (h+0.5f) - 0.5f)*stride;
                    // Python code suggests kpt score might not strictly need Sigmoid if already processed?
                    // But standard YOLO head outputs raw logits. Let's apply Sigmoid just in case.
                    // If results are bad, try raw value.
                    det.kpt_scores[k] = 1.0f/(1.0f+std::exp(-kc));
                }
                dets.push_back(det);
            }
        }
    }

    // NMS
    std::vector<int> indices;
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    for(auto& d : dets) { boxes.push_back(d.bbox); scores.push_back(d.score); }
    cv::dnn::NMSBoxes(boxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    auto dur_post = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_post).count() / 1000.0;
    LOG_TIME("Post-processing time", dur_post);
    LOG_INFO("Detected: " << indices.size() << " objects");

    // 8. Draw and Save
    std::vector<PoseDetection> final_dets;
    for(int idx : indices) final_dets.push_back(dets[idx]);
    
    if(!final_dets.empty()) {
        draw_pose(img, final_dets, x_scale, y_scale, x_shift, y_shift);
    }
    
    cv::imwrite(save_path, img);
    LOG_INFO("Saved result to " << save_path);

    // 9. Cleanup
    hbUCPReleaseTask(task);
    for(auto& t : input_tensors) hbUCPFree(&t.sysMem);
    for(auto& o : outputs) hbUCPFree(&o.sysMem);
    hbDNNRelease(packed_handle);
    return 0;
}
