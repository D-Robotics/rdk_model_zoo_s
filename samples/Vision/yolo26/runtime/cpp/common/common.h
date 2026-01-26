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
 * @file common.h
 * @brief YOLO26 C++ inference utilities
 */

#ifndef YOLO26_COMMON_H
#define YOLO26_COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#ifdef _WIN32
#include <direct.h>
#define getcwd _getcwd
#else
#include <unistd.h>
#include <libgen.h>
#endif

#define CHECK_SUCCESS(value, errmsg)                                         \
    do {                                                                     \
        auto ret_code = value;                                               \
        if (ret_code != 0) {                                                 \
            std::cerr << "\033[1;31m[ERROR]\033[0m " << __FILE__ << ":"      \
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
    std::cout << "\033[1;36m[" << msg << "]\033[0m = " << std::fixed         \
              << std::setprecision(2) << (duration) << " ms" << std::endl

// ============================================================================
// Detection Structure
// ============================================================================

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2d bbox;
    
    Detection(int id, float conf, const cv::Rect2d& box) 
        : class_id(id), confidence(conf), bbox(box) {}
};

// ============================================================================
// Path Utilities
// ============================================================================

/**
 * @brief Get project root directory relative to current working directory

 * build -> detect -> cpp -> runtime -> yolo26 -> Vision -> samples -> root
 */
inline std::string getProjectRoot() {
    char cwd[4096];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        return ".";
    }
    std::string relative_path = std::string(cwd) + "/../../../../../../../";
    
    char* resolved = realpath(relative_path.c_str(), nullptr);
    if (resolved) {
        std::string result(resolved);
        free(resolved);
        return result;
    }
    return relative_path;
}

/**
 * @brief Get default label file path based on task type
 */
inline std::string getDefaultLabelPath(const std::string& project_root, 
                                        const std::string& task) {
    if (task == "detect" || task == "seg" || task == "pose") {
        return project_root + "/datasets/coco/coco_classes.names";
    } else if (task == "cls") {
        return project_root + "/datasets/imagenet/imagenet_classes.names";
    } else if (task == "obb") {
        return project_root + "/datasets/dotav1/dota_classes.names";
    }
    return project_root + "/datasets/coco/coco_classes.names";
}

// ============================================================================
// Default Color Palette for Visualization
// ============================================================================

inline const std::vector<cv::Scalar>& getDefaultColors() {
    static const std::vector<cv::Scalar> COLORS = {
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
    return COLORS;
}

// ============================================================================
// Class Names Loading
// ============================================================================

inline std::vector<std::string> loadClassNames(const std::string& filepath) {
    std::vector<std::string> names;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return names;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
            line.pop_back();
        }
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    std::cout << "Loaded default labels from " << filepath << std::endl;
    return names;
}

// ============================================================================
// Preprocessing Utilities
// ============================================================================

/**
 * @brief Preprocess image with resize or letterbox
 */
inline cv::Mat preprocessImage(const cv::Mat& img, int input_h, int input_w,
                                float& x_scale, float& y_scale,
                                int& x_shift, int& y_shift,
                                bool letterbox = true) {
    cv::Mat result;
    
    if (letterbox) {
        x_scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        y_scale = x_scale;
        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);
        
        x_shift = (input_w - new_w) / 2;
        y_shift = (input_h - new_h) / 2;
        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, 
                          y_shift, input_h - new_h - y_shift, 
                          x_shift, input_w - new_w - x_shift, 
                          cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else {
        cv::resize(img, result, cv::Size(input_w, input_h));
        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
        x_shift = 0;
        y_shift = 0;
    }
    
    return result;
}

/**
 * @brief Convert BGR image to NV12 format
 */
inline cv::Mat bgr2nv12(const cv::Mat& bgr_img) {
    int h = bgr_img.rows, w = bgr_img.cols;
    cv::Mat yuv;
    cv::cvtColor(bgr_img, yuv, cv::COLOR_BGR2YUV_I420);
    cv::Mat nv12(h * 3 / 2, w, CV_8UC1);
    uint8_t* y_ptr = nv12.ptr<uint8_t>();
    uint8_t* uv_ptr = y_ptr + h * w;
    uint8_t* u_src = yuv.ptr<uint8_t>() + h * w;
    uint8_t* v_src = u_src + (h/2) * (w/2);
    
    memcpy(y_ptr, yuv.ptr<uint8_t>(), h * w);
    
    for (int i = 0; i < (h/2) * (w/2); ++i) {
        *uv_ptr++ = *u_src++;
        *uv_ptr++ = *v_src++;
    }

    return nv12;
}

// ============================================================================
// Print Detection Report
// ============================================================================

/**
 * @brief Print detection results
 */
inline void printDetections(const std::vector<Detection>& detections,
                            const std::vector<std::string>& class_names,
                            float x_scale, float y_scale,
                            int x_shift, int y_shift) {
    std::cout << "\n" << std::string(20, '=') << " Detection Report " 
              << std::string(20, '=') << std::endl;
    std::cout << "Total Objects Found: " << detections.size() << std::endl;
    
    if (detections.empty()) {
        std::cout << "No objects detected." << std::endl;
        return;
    }
    
    std::cout << std::left << std::setw(5) << "ID" 
              << std::setw(16) << "Label"
              << std::setw(9) << "Score"
              << "Box (x1, y1, x2, y2)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        
        int x1 = static_cast<int>((det.bbox.x - x_shift) / x_scale);
        int y1 = static_cast<int>((det.bbox.y - y_shift) / y_scale);
        int x2 = static_cast<int>((det.bbox.x + det.bbox.width - x_shift) / x_scale);
        int y2 = static_cast<int>((det.bbox.y + det.bbox.height - y_shift) / y_scale);
        
        std::string label;
        if (det.class_id < static_cast<int>(class_names.size())) {
            label = class_names[det.class_id];
        } else {
            label = std::to_string(det.class_id);
        }
        
        std::cout << std::left << std::setw(5) << i
                  << std::setw(16) << label
                  << std::fixed << std::setprecision(2) << std::setw(9) << det.confidence
                  << "[" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]"
                  << std::endl;
    }
    
    std::cout << std::string(58, '=') << std::endl;
}

// ============================================================================
// Visualization Utilities
// ============================================================================

/**
 * @brief Draw detection results on image
 */
inline void drawDetections(cv::Mat& img, 
                           const std::vector<Detection>& detections,
                           const std::vector<std::string>& class_names,
                           float x_scale, float y_scale, 
                           int x_shift, int y_shift,
                           double font_scale = 0.6,
                           int font_thickness = 2,
                           int box_thickness = 2) {
    const auto& colors = getDefaultColors();

    for (const auto& det : detections) {
        float x1 = (det.bbox.x - x_shift) / x_scale;
        float y1 = (det.bbox.y - y_shift) / y_scale;
        float x2 = x1 + det.bbox.width / x_scale;
        float y2 = y1 + det.bbox.height / y_scale;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img.cols)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img.rows)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img.cols)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img.rows)));

        cv::Scalar color = colors[det.class_id % colors.size()];

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, box_thickness);

        std::string label;
        if (det.class_id < static_cast<int>(class_names.size())) {
            label = class_names[det.class_id];
        } else {
            label = "class_" + std::to_string(det.class_id);
        }
        label += ": " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             font_scale, font_thickness, &baseline);

        int label_y = std::max(static_cast<int>(y1), text_size.height + 10);
        cv::rectangle(img,
                     cv::Point(x1, label_y - text_size.height - 10),
                     cv::Point(x1 + text_size.width, label_y),
                     color, cv::FILLED);

        cv::putText(img, label, cv::Point(x1, label_y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, font_scale,
                   cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
    }
}

// ============================================================================
// YOLO26 Anchor-Free Box Decoding
// ============================================================================

/**
 * @brief Sigmoid activation function
 */
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Calculate logit threshold from sigmoid threshold
 * 
 * sigmoid(x) >= thres  <=>  x >= -ln(1/thres - 1)
 */
inline float calcLogitThreshold(float thres) {
    float safe_thres = std::max(1e-6f, std::min(thres, 1.0f - 1e-6f));
    return -std::log(1.0f / safe_thres - 1.0f);
}

#endif // YOLO26_COMMON_H
