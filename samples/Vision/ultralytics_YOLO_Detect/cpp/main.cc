/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2025，SkyXZ D-Robotics.

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

// 注意: 此程序在RDK S100板端运行
// Attention: This program runs on RDK S100 board.

// D-Robotics S100 *.hbm 模型路径
// Path of D-Robotics S100 *.hbm model.
#define MODEL_PATH "rdk_model_zoo_s/samples/Vision/ultralytics_YOLO_Detect/source/reference_hbm_models/yolov5nu_detect_nashe_640x640_nv12.hbm"

// 推理使用的测试图片路径
// Path of the test image used for inference.
#define TEST_IMG_PATH "rdk_model_zoo_s/resource/datasets/COCO2017/assets/bus.jpg"

// 前处理方式选择, 0:Resize, 1:LetterBox
// Preprocessing method selection, 0: Resize, 1: LetterBox
#define RESIZE_TYPE 0 
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 推理结果保存路径
// Path where the inference result will be saved
#define IMG_SAVE_PATH "cpp_result.jpg"

// 模型的类别数量, 默认80
// Number of classes in the model, default is 80
#define CLASSES_NUM 80

// NMS的阈值, 默认0.7
// Non-Maximum Suppression (NMS) threshold, default is 0.7
#define NMS_THRESHOLD 0.7

// 分数阈值, 默认0.25
// Score threshold, default is 0.25
#define SCORE_THRESHOLD 0.25

// 控制回归部分离散化程度的超参数, 默认16
// A hyperparameter that controls the discretization level of the regression part, default is 16
#define REG 16

// 绘制标签的字体尺寸, 默认1.0
// Font size for drawing labels, default is 1.0.
#define FONT_SIZE 1.0

// 绘制标签的字体粗细, 默认 1.0
// Font thickness for drawing labels, default is 1.0.
#define FONT_THICKNESS 1.0

// 绘制矩形框的线宽, 默认2.0
// Line width for drawing bounding boxes, default is 2.0.
#define LINE_SIZE 2.0

// C/C++ Standard Libraries
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

// Third Party Libraries
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// RDK S100 UCP API
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#define RDK_CHECK_SUCCESS(value, errmsg)                                         \
    do                                                                           \
    {                                                                            \
        auto ret_code = value;                                                   \
        if (ret_code != 0)                                                       \
        {                                                                        \
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cout << errmsg << ", error code:" << ret_code << std::endl;     \
            return ret_code;                                                     \
        }                                                                        \
    } while (0);

// COCO Names
std::vector<std::string> object_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

// S100定制颜色
std::vector<cv::Scalar> rdk_colors = {
    cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
    cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
    cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132),
    cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)
};

// Softmax function for DFL calculation
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

int main()
{
    // Step 0: 加载S100 hbm模型
    // Step 0: Load S100 hbm model
    auto begin_time = std::chrono::system_clock::now();
    hbDNNPackedHandle_t packed_dnn_handle;
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");
    std::cout << "\033[31m Load D-Robotics S100 Quantize model time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;

    // Step 1: 打印基本信息
    // Step 1: Print basic information
    std::cout << "[INFO] OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "[INFO] MODEL_PATH: " << MODEL_PATH << std::endl;
    std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
    std::cout << "[INFO] NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
    std::cout << "[INFO] SCORE_THRESHOLD: " << SCORE_THRESHOLD << std::endl;

    // Step 2: 获取模型句柄
    // Step 2: Get model handle
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");

    if (model_count > 1) {
        std::cout << "This model file have more than 1 model, only use model 0." << std::endl;
    }
    const char *model_name = model_name_list[0];
    std::cout << "[model name]: " << model_name << std::endl;

    hbDNNHandle_t dnn_handle;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // Step 3: 检查模型输入
    // Step 3: Check model input
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");

    if (input_count < 1) {
        std::cout << "S100 YOLO model should have at least 1 input, but got " << input_count << std::endl;
        return -1;
    } else if (input_count > 1) {
        std::cout << "S100 YOLO model has " << input_count << " inputs, using first input for inference" << std::endl;
    } 

    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");

    // S100 UCP 模型需要检测输入格式是否支持
    std::cout << "✓ input tensor type: " << input_properties.tensorType << std::endl;
    
    // 检测输入格式是否为NV12 (type 3)
    if (input_properties.tensorType != 3) {
        std::cout << "[ERROR] This program only supports NV12 input (type 3), but got type: " << input_properties.tensorType << std::endl;
        return -1;
    }

    // 检测输入tensor布局为NCHW
    if (input_properties.validShape.numDimensions == 4) {
        // NCHW布局，H和W应该在维度1和2位置，且通道数应该为1
        int32_t channels = input_properties.validShape.dimensionSize[3];
        if (channels != 1) {
            std::cout << "[ERROR] This program expects NCHW layout with 1 channel, but got " << channels << " channels" << std::endl;
            return -1;
        }
        std::cout << "✓ input tensor layout: NCHW (verified)" << std::endl;
    } else {
        std::cout << "[ERROR] Expected 4D input tensor for NCHW layout, but got " << input_properties.validShape.numDimensions << "D" << std::endl;
        return -1;
    }

    // 获取输入尺寸
    int32_t input_H, input_W;
    if (input_properties.validShape.numDimensions == 4) {
        input_H = input_properties.validShape.dimensionSize[1];
        input_W = input_properties.validShape.dimensionSize[2];
        std::cout << "✓ input tensor valid shape: (" 
                  << input_properties.validShape.dimensionSize[0] << ", "
                  << input_H << ", " << input_W << ", "
                  << input_properties.validShape.dimensionSize[3] << ")" << std::endl;
    } else {
        std::cout << "S100 YOLO model input should be 4D" << std::endl;
        return -1;
    }

    // Step 4: 检查模型输出 - S100 YOLO 按照Readme导出后应该有6个输出
    // Step 4: Check model output - S100 YOLO should have 6 outputs according to Readme
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");

    if (output_count != 6) {
        std::cout << "S100 YOLO model should have 6 outputs, but got " << output_count << std::endl;
        return -1;
    }
    std::cout << "✓ S100 YOLO model has 6 outputs" << std::endl;

    // 打印输出信息并获取正确的输出顺序
    std::cout << "\033[32m-> output tensors\033[0m" << std::endl;
    for (int i = 0; i < 6; i++) {
        hbDNNTensorProperties output_properties;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
        std::cout << "output[" << i << "] valid shape: (" 
                  << output_properties.validShape.dimensionSize[0] << ", "
                  << output_properties.validShape.dimensionSize[1] << ", "
                  << output_properties.validShape.dimensionSize[2] << ", "
                  << output_properties.validShape.dimensionSize[3] << "), ";
        
        std::cout << "QuantiType: " << output_properties.quantiType << std::endl;
    }

    // Step 5: 前处理 - 读取图像并转换为YUV420SP
    // Step 5: Preprocessing - Load image and convert to YUV420SP
    std::cout << "\033[32m-> Starting preprocessing\033[0m" << std::endl;
    cv::Mat img = cv::imread(TEST_IMG_PATH);
    if (img.empty()) {
        std::cout << "Failed to load image: " << TEST_IMG_PATH << std::endl;
        return -1;
    }
    std::cout << "✓ img path: " << TEST_IMG_PATH << std::endl;
    std::cout << "✓ img (rows, cols, channels): (" << img.rows << ", " << img.cols << ", " << img.channels() << ")" << std::endl;

    // 前处理参数
    float y_scale = 1.0, x_scale = 1.0;
    int x_shift = 0, y_shift = 0;
    cv::Mat resize_img;

    begin_time = std::chrono::system_clock::now();
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        // LetterBox前处理
        float scale = std::min(1.0f * input_H / img.rows, 1.0f * input_W / img.cols);
        
        int new_w = int(img.cols * scale);
        int new_h = int(img.rows * scale);
        
        // 确保尺寸为偶数
        new_w = (new_w / 2) * 2;
        new_h = (new_h / 2) * 2;
        
        // 重新计算实际的缩放因子
        x_scale = 1.0f * new_w / img.cols;
        y_scale = 1.0f * new_h / img.rows;
        
        x_shift = (input_W - new_w) / 2;
        int x_other = input_W - new_w - x_shift;
        
        y_shift = (input_H - new_h) / 2;
        int y_other = input_H - new_h - y_shift;
        
        cv::Size targetSize(new_w, new_h);
        cv::resize(img, resize_img, targetSize);
        cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else {
        // Resize前处理
        cv::Size targetSize(input_W, input_H);
        cv::resize(img, resize_img, targetSize);
        y_scale = 1.0 * input_H / img.rows;
        x_scale = 1.0 * input_W / img.cols;
    }

    std::cout << "✓ y_scale = " << y_scale << ", x_scale = " << x_scale << std::endl;
    std::cout << "✓ y_shift = " << y_shift << ", x_shift = " << x_shift << std::endl;

    // BGR转YUV420SP (NV12)
    cv::Mat img_nv12;
    cv::Mat yuv_mat;
    cv::cvtColor(resize_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *yuv = yuv_mat.ptr<uint8_t>();
    
    img_nv12 = cv::Mat(input_H * 3 / 2, input_W, CV_8UC1);
    uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
    int uv_height = input_H / 2;
    int uv_width = input_W / 2;
    int y_size = input_H * input_W;
    
    // 复制Y平面
    memcpy(ynv12, yuv, y_size);
    
    // 交错UV平面
    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;
    for (int i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }

    std::cout << "\033[31m pre process time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;

    // Step 6: 准备输入tensor
    // Step 6: Prepare input tensor
    std::vector<hbDNNTensor> input_tensors(input_count);
    std::vector<hbDNNTensor> output_tensors(output_count);

    // 分配输入内存
    for (int i = 0; i < input_count; i++) {
        // 复制输入tensor属性
        input_tensors[i].properties = input_properties;
        
        int data_size;
        if (i == 0) {
            // 第一个输入：Y分量 640x640x1
            data_size = input_H * input_W;
            
            // 设置tensor的stride信息
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = input_H;
            input_tensors[i].properties.validShape.dimensionSize[2] = input_W;
            input_tensors[i].properties.validShape.dimensionSize[3] = 1;
            
            // 设置stride 
            input_tensors[i].properties.stride[3] = 1;                    // 每个元素1字节
            input_tensors[i].properties.stride[2] = 1;                    // 通道步长 = stride[3] * size[3] = 1 * 1
            input_tensors[i].properties.stride[1] = input_W;              // 行步长 = stride[2] * size[2] = 1 * 640 = 640
            input_tensors[i].properties.stride[0] = input_W * input_H;    // 整个tensor = stride[1] * size[1] = 640 * 640 = 409600
        } else {
            // 第二个输入：UV分量 320x320x2 (尺寸减半，2通道)
            int uv_h = input_H / 2;  // 320
            int uv_w = input_W / 2;  // 320
            data_size = uv_h * uv_w * 2;  // UV两个通道
            
            // 设置tensor的stride信息
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = uv_h;
            input_tensors[i].properties.validShape.dimensionSize[2] = uv_w; 
            input_tensors[i].properties.validShape.dimensionSize[3] = 2;
            
            // 设置stride
            input_tensors[i].properties.stride[3] = 1;                    // 每个元素1字节
            input_tensors[i].properties.stride[2] = 2;                    // 通道步长 = stride[3] * size[3] = 1 * 2 = 2
            input_tensors[i].properties.stride[1] = uv_w * 2;             // 行步长 = stride[2] * size[2] = 2 * 320 = 640
            input_tensors[i].properties.stride[0] = uv_w * uv_h * 2;      // 整个tensor = stride[1] * size[1] = 640 * 320 = 204800
        }
        
        // 分配内存
        hbUCPMallocCached(&input_tensors[i].sysMem, data_size, 0);
        
        std::cout << "✓ Input tensor " << i << " memory allocated: " << data_size << " bytes" << std::endl;
        
        // 复制数据
        if (i == 0) {
            // 第一个输入：复制Y分量
            memcpy(input_tensors[i].sysMem.virAddr, ynv12, input_H * input_W);
            std::cout << "✓ Y component data copied to tensor " << i << std::endl;
        } else {
            // 第二个输入：复制UV分量 
            uint8_t *uv_src = ynv12 + input_H * input_W;  // UV数据在Y之后
            memcpy(input_tensors[i].sysMem.virAddr, uv_src, data_size);
            std::cout << "✓ UV component data copied to tensor " << i << std::endl;
        }
        
        // 刷新内存
        hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    // 分配输出内存
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties &output_properties = output_tensors[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbUCPSysMem &mem = output_tensors[i].sysMem;
        hbUCPMallocCached(&mem, out_aligned_size, 0);
        std::cout << "✓ Output tensor " << i << " memory allocated: " << out_aligned_size << " bytes" << std::endl;
    }

    // Step 7: 推理
    // Step 7: Inference
    std::cout << "\033[32m-> Starting inference\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now();
    
    // 生成任务句柄
    hbUCPTaskHandle_t task_handle = nullptr;
    int infer_ret = hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle);
    if (infer_ret != 0) {
        std::cout << "[ERROR] hbDNNInferV2 failed with error code: " << infer_ret << std::endl;
        return -1;
    }
    
    if (task_handle == nullptr) {
        std::cout << "[ERROR] task_handle is null after hbDNNInferV2" << std::endl;
        return -1;
    }
    
    std::cout << "✓ Inference task created successfully" << std::endl;
    
    // 设置UCP调度参数
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;  // 使用任意BPU核心
    
    // 提交任务到UCP
    int submit_ret = hbUCPSubmitTask(task_handle, &ctrl_param);
    if (submit_ret != 0) {
        std::cout << "[ERROR] hbUCPSubmitTask failed with error code: " << submit_ret << std::endl;
        return -1;
    }
    
    std::cout << "✓ Inference task submitted successfully" << std::endl;
    
    // 等待任务完成，设置合理的超时时间(10秒)
    int wait_ret = hbUCPWaitTaskDone(task_handle, 10000);
    if (wait_ret != 0) {
        std::cout << "[ERROR] hbUCPWaitTaskDone failed with error code: " << wait_ret << std::endl;
        return -1;
    }
    
    std::cout << "✓ Inference task completed successfully" << std::endl;
    
    std::cout << "\033[31m forward time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;

    // Step 8: 后处理
    // Step 8: Post-processing
    std::cout << "\033[32m-> Starting post-processing\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now();

    // 计算置信度阈值的原始值（利用Sigmoid函数的单调性）
    float CONF_THRES_RAW = -std::log(1.0f / SCORE_THRESHOLD - 1.0f);
    
    // 预计算锚点
    // s_anchor: 80x80 (stride=8)
    std::vector<std::pair<float, float>> s_anchor(80 * 80);
    for (int h = 0; h < 80; h++) {
        for (int w = 0; w < 80; w++) {
            s_anchor[h * 80 + w] = {w + 0.5f, h + 0.5f};
        }
    }
    
    // m_anchor: 40x40 (stride=16) 
    std::vector<std::pair<float, float>> m_anchor(40 * 40);
    for (int h = 0; h < 40; h++) {
        for (int w = 0; w < 40; w++) {
            m_anchor[h * 40 + w] = {w + 0.5f, h + 0.5f};
        }
    }
    
    // l_anchor: 20x20 (stride=32)
    std::vector<std::pair<float, float>> l_anchor(20 * 20);
    for (int h = 0; h < 20; h++) {
        for (int w = 0; w < 20; w++) {
            l_anchor[h * 20 + w] = {w + 0.5f, h + 0.5f};
        }
    }

    // 处理3个特征层的输出
    std::vector<cv::Rect2d> all_bboxes;
    std::vector<float> all_scores;
    std::vector<int> all_ids;
    
    // 处理每个尺度
    for (int scale = 0; scale < 3; scale++) {
        int cls_idx = scale * 2;     // 0, 2, 4
        int bbox_idx = scale * 2 + 1; // 1, 3, 5
        int stride = (scale == 0) ? 8 : (scale == 1) ? 16 : 32;
        int grid_size = (scale == 0) ? 80 : (scale == 1) ? 40 : 20;
        
        // 刷新BPU内存
        hbUCPMemFlush(&(output_tensors[cls_idx].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
        hbUCPMemFlush(&(output_tensors[bbox_idx].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
        
        // 获取输出数据指针
        auto *cls_data = reinterpret_cast<float *>(output_tensors[cls_idx].sysMem.virAddr);
        auto *bbox_data = reinterpret_cast<int32_t *>(output_tensors[bbox_idx].sysMem.virAddr);
        auto *bbox_scale = reinterpret_cast<float *>(output_tensors[bbox_idx].properties.scale.scaleData);
        
        int total_anchors = grid_size * grid_size;
        
        // 第一步：找到所有超过阈值的位置
        std::vector<int> valid_indices;
        std::vector<int> valid_class_ids;
        std::vector<float> valid_scores;
        
        for (int i = 0; i < total_anchors; i++) {
            float *cur_cls = cls_data + i * CLASSES_NUM;
            
            // 找到最大分数和对应类别
            int max_cls_id = 0;
            for (int c = 1; c < CLASSES_NUM; c++) {
                if (cur_cls[c] > cur_cls[max_cls_id]) {
                    max_cls_id = c;
                }
            }
            
            // 检查是否超过阈值（raw值比较）
            if (cur_cls[max_cls_id] >= CONF_THRES_RAW) {
                valid_indices.push_back(i);
                valid_class_ids.push_back(max_cls_id);
                // 计算Sigmoid分数
                float score = 1.0f / (1.0f + std::exp(-cur_cls[max_cls_id]));
                valid_scores.push_back(score);
            }
        }
        
        // 第二步：处理有效检测的边界框
        for (size_t idx = 0; idx < valid_indices.size(); idx++) {
            int anchor_idx = valid_indices[idx];
            int32_t *cur_bbox = bbox_data + anchor_idx * (REG * 4);
            
            // DFL计算 - 对每条边进行处理
            float ltrb[4];
            for (int i = 0; i < 4; i++) {
                float dfl_values[REG];
                float softmax_values[REG];
                
                // 反量化DFL值
                for (int j = 0; j < REG; j++) {
                    int scale_idx = i * REG + j;
                    dfl_values[j] = float(cur_bbox[scale_idx]) * bbox_scale[scale_idx];
                }
                
                // Softmax
                softmax(dfl_values, softmax_values, REG);
                
                // 计算期望值（DFL到距离的转换）
                ltrb[i] = 0.0f;
                for (int j = 0; j < REG; j++) {
                    ltrb[i] += softmax_values[j] * j;
                }
            }
            
            // 获取锚点坐标
            float anchor_x, anchor_y;
            if (scale == 0) {
                anchor_x = s_anchor[anchor_idx].first;
                anchor_y = s_anchor[anchor_idx].second;
            } else if (scale == 1) {
                anchor_x = m_anchor[anchor_idx].first;
                anchor_y = m_anchor[anchor_idx].second;
            } else {
                anchor_x = l_anchor[anchor_idx].first;
                anchor_y = l_anchor[anchor_idx].second;
            }
            
            // ltrb转xyxy坐标
            double x1 = (anchor_x - ltrb[0]) * stride;
            double y1 = (anchor_y - ltrb[1]) * stride;
            double x2 = (anchor_x + ltrb[2]) * stride;
            double y2 = (anchor_y + ltrb[3]) * stride;
            
            // 检查边界框合法性
            if (x2 > x1 && y2 > y1) {
                all_bboxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                all_scores.push_back(valid_scores[idx]);
                all_ids.push_back(valid_class_ids[idx]);
            }
        }
    }

    // Step 9: 分类别NMS处理
    // Step 9: Class-wise NMS processing
    std::vector<std::vector<int>> nms_indices(CLASSES_NUM);
    int total_detections_before_nms = all_bboxes.size();
    int total_detections_after_nms = 0;
    
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
        // 收集该类别的所有检测
        std::vector<cv::Rect2d> class_bboxes;
        std::vector<float> class_scores;
        std::vector<int> original_indices;
        
        for (size_t i = 0; i < all_bboxes.size(); i++) {
            if (all_ids[i] == cls_id) {
                class_bboxes.push_back(all_bboxes[i]);
                class_scores.push_back(all_scores[i]);
                original_indices.push_back(i);
            }
        }
        
        if (!class_bboxes.empty()) {
            std::vector<int> class_nms_indices;
            cv::dnn::NMSBoxes(class_bboxes, class_scores, 
                             SCORE_THRESHOLD, NMS_THRESHOLD, class_nms_indices);
            
            // 将类别内的索引转换为全局索引
            for (int idx : class_nms_indices) {
                nms_indices[cls_id].push_back(original_indices[idx]);
            }
            total_detections_after_nms += class_nms_indices.size();
        }
    }
    
    std::cout << "✓ Detections before NMS: " << total_detections_before_nms << std::endl;
    std::cout << "✓ Detections after NMS: " << total_detections_after_nms << std::endl;

    std::cout << "\033[31m Post Process time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;

    // Step 10: 绘制结果
    // Step 10: Draw results
    std::cout << "\033[32m-> Drawing results\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now();
    
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
        for (int global_idx : nms_indices[cls_id]) {
            // 坐标转换回原图
            float x1 = (all_bboxes[global_idx].x - x_shift) / x_scale;
            float y1 = (all_bboxes[global_idx].y - y_shift) / y_scale;
            float x2 = x1 + all_bboxes[global_idx].width / x_scale;
            float y2 = y1 + all_bboxes[global_idx].height / y_scale;
            float score = all_scores[global_idx];
            
            // 边界检查
            x1 = std::max(0.0f, std::min((float)img.cols - 1, x1));
            y1 = std::max(0.0f, std::min((float)img.rows - 1, y1));
            x2 = std::max(0.0f, std::min((float)img.cols - 1, x2));
            y2 = std::max(0.0f, std::min((float)img.rows - 1, y2));
            
            // 绘制边界框
            cv::Scalar color = rdk_colors[cls_id % 20];
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, LINE_SIZE);
            
            // 绘制标签
            std::string label = object_names[cls_id] + ": " + std::to_string(int(score * 100)) + "%";
            int baseline;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_THICKNESS, &baseline);
            
            cv::Point label_pos(x1, y1 - 10 > textSize.height ? y1 - 10 : y1 + textSize.height + 10);
            cv::rectangle(img, label_pos + cv::Point(0, baseline), 
                         label_pos + cv::Point(textSize.width, -textSize.height), color, cv::FILLED);
            cv::putText(img, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 0), FONT_THICKNESS);
            
            // 打印检测结果
            std::cout << "(" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ") -> " 
                      << object_names[cls_id] << ": " << std::fixed << std::setprecision(2) << score << std::endl;
        }
    }
    
    std::cout << "\033[31m Draw Result time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;

    // Step 11: 保存结果
    // Step 11: Save result
    cv::imwrite(IMG_SAVE_PATH, img);
    std::cout << "\033[32m✓ saved in path: \"" << IMG_SAVE_PATH << "\"\033[0m" << std::endl;

    // Step 12: 资源释放
    // Step 12: Release resources
    std::cout << "\033[32m-> Cleaning up resources\033[0m" << std::endl;
    
    // 释放任务句柄
    hbUCPReleaseTask(task_handle);
    
    // 释放输入内存
    for (int i = 0; i < input_count; i++) {
        hbUCPFree(&(input_tensors[i].sysMem));
    }
    
    // 释放输出内存
    for (int i = 0; i < output_count; i++) {
        hbUCPFree(&(output_tensors[i].sysMem));
    }
    
    // 释放模型
    hbDNNRelease(packed_dnn_handle);
    
    std::cout << "\033[32m✓ Program completed successfully\033[0m" << std::endl;
    return 0;
}
