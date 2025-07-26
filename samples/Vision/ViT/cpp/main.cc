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
// D-Robotics S100 ViT *.hbm 模型路径
// Path of D-Robotics S100 ViT *.hbm model.
#define MODEL_PATH "rdk_model_zoo_s/samples/Vision/ViT/cpp/vit_cifar10_batch1.hbm"
// 推理使用的测试图片路径
// Path of the test image used for inference.
#define TEST_IMG_PATH "rdk_model_zoo_s/samples/Vision/ViT/source/imgs/airplane_0000.png"
// 前处理方式选择, 0:Resize, 1:Center Crop
// Preprocessing method selection, 0: Resize, 1: Center Crop
#define RESIZE_TYPE 0 
#define CENTER_CROP_TYPE 1
#define PREPROCESS_TYPE RESIZE_TYPE
// 模型输入尺寸 (ViT通常使用224x224)
// Model input size (ViT typically uses 224x224)
#define INPUT_HEIGHT 224
#define INPUT_WIDTH 224
// 模型的类别数量 (ImageNet 1000类, CIFAR-10 10类)
// Number of classes in the model (ImageNet 1000 classes, CIFAR-10 10 classes)
#define CLASSES_NUM 10
// 分类结果的Top-K数量
// Top-K number for classification results
#define TOP_K 5
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
// CIFAR-10类别名称
std::vector<std::string> imagenet_classes = {
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
};
// Softmax函数
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
    std::cout << "[INFO] TOP_K: " << TOP_K << std::endl;
    std::cout << "[INFO] INPUT_HEIGHT: " << INPUT_HEIGHT << std::endl;
    std::cout << "[INFO] INPUT_WIDTH: " << INPUT_WIDTH << std::endl;
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
        std::cout << "S100 ViT model should have at least 1 input, but got " << input_count << std::endl;
        return -1;
    } else if (input_count > 1) {
        std::cout << "S100 ViT model has " << input_count << " inputs, using first input for inference" << std::endl;
    } 
    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");
    // 检测输入格式
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
        std::cout << "S100 ViT model input should be 4D" << std::endl;
        return -1;
    }
    // Step 4: 检查模型输出 - S100 ViT 按照Readme导出后应该有1个输出
    // Step 4: Check model output - S100 ViT should have 1 output according to Readme
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");
    if (output_count != 1) {
        std::cout << "S100 ViT model should have 1 output, but got " << output_count << std::endl;
        return -1;
    }
    std::cout << "✓ S100 ViT model has 1 output" << std::endl;
    // 打印输出信息并获取正确的输出顺序
    std::cout << "\033[32m-> output tensors\033[0m" << std::endl;
    for (int i = 0; i < 1; i++) {
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
    if (PREPROCESS_TYPE == CENTER_CROP_TYPE) {
        // Center Crop前处理
        int new_h = INPUT_HEIGHT;
        int new_w = INPUT_WIDTH;
        // 确保尺寸为偶数
        new_w = (new_w / 2) * 2;
        new_h = (new_h / 2) * 2;
        // 重新计算实际的缩放因子
        x_scale = 1.0f * new_w / img.cols;
        y_scale = 1.0f * new_h / img.rows;
        x_shift = (INPUT_WIDTH - new_w) / 2;
        int x_other = INPUT_WIDTH - new_w - x_shift;
        y_shift = (INPUT_HEIGHT - new_h) / 2;
        int y_other = INPUT_HEIGHT - new_h - y_shift;
        cv::Size targetSize(new_w, new_h);
        cv::resize(img, resize_img, targetSize);
        cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else {
        // Resize前处理
        cv::Size targetSize(INPUT_WIDTH, INPUT_HEIGHT);
        cv::resize(img, resize_img, targetSize);
        y_scale = 1.0 * INPUT_HEIGHT / img.rows;
        x_scale = 1.0 * INPUT_WIDTH / img.cols;
    }
    std::cout << "✓ y_scale = " << y_scale << ", x_scale = " << x_scale << std::endl;
    std::cout << "✓ y_shift = " << y_shift << ", x_shift = " << x_shift << std::endl;
    // BGR转YUV420SP (NV12)
    cv::Mat img_nv12;
    cv::Mat yuv_mat;
    cv::cvtColor(resize_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *yuv = yuv_mat.ptr<uint8_t>();
    img_nv12 = cv::Mat(INPUT_HEIGHT * 3 / 2, INPUT_WIDTH, CV_8UC1);
    uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
    int uv_height = INPUT_HEIGHT / 2;
    int uv_width = INPUT_WIDTH / 2;
    int y_size = INPUT_HEIGHT * INPUT_WIDTH;
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
            // 第一个输入：Y分量 224x224x1
            data_size = INPUT_HEIGHT * INPUT_WIDTH;
            // 设置tensor的stride信息
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = INPUT_HEIGHT;
            input_tensors[i].properties.validShape.dimensionSize[2] = INPUT_WIDTH;
            input_tensors[i].properties.validShape.dimensionSize[3] = 1;
            // 设置stride 
            input_tensors[i].properties.stride[3] = 1;                    // 每个元素1字节
            input_tensors[i].properties.stride[2] = 1;                    // 通道步长 = stride[3] * size[3] = 1 * 1
            input_tensors[i].properties.stride[1] = INPUT_WIDTH;          // 行步长 = stride[2] * size[2] = 1 * 224
            input_tensors[i].properties.stride[0] = INPUT_WIDTH * INPUT_HEIGHT;    // 整个tensor = stride[1] * size[1] = 224 * 224
        } else {
            // 第二个输入：UV分量 112x112x2 (尺寸减半，2通道)
            int uv_h = INPUT_HEIGHT / 2;  // 112
            int uv_w = INPUT_WIDTH / 2;   // 112
            data_size = uv_h * uv_w * 2;  // UV两个通道
            // 设置tensor的stride信息
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = uv_h;
            input_tensors[i].properties.validShape.dimensionSize[2] = uv_w; 
            input_tensors[i].properties.validShape.dimensionSize[3] = 2;
            // 设置stride
            input_tensors[i].properties.stride[3] = 1;                    // 每个元素1字节
            input_tensors[i].properties.stride[2] = 2;                    // 通道步长 = stride[3] * size[3] = 1 * 2 = 2
            input_tensors[i].properties.stride[1] = uv_w * 2;             // 行步长 = stride[2] * size[2] = 2 * 112 = 224
            input_tensors[i].properties.stride[0] = uv_w * uv_h * 2;      // 整个tensor = stride[1] * size[1] = 224 * 112
        }
        // 分配内存
        hbUCPMallocCached(&input_tensors[i].sysMem, data_size, 0);
        std::cout << "✓ Input tensor " << i << " memory allocated: " << data_size << " bytes" << std::endl;
        // 复制数据
        if (i == 0) {
            // 第一个输入：复制Y分量
            memcpy(input_tensors[i].sysMem.virAddr, ynv12, INPUT_HEIGHT * INPUT_WIDTH);
            std::cout << "✓ Y component data copied to tensor " << i << std::endl;
        } else {
            // 第二个输入：复制UV分量 
            uint8_t *uv_src = ynv12 + INPUT_HEIGHT * INPUT_WIDTH;  // UV数据在Y之后
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
    // 刷新BPU内存
    hbUCPMemFlush(&(output_tensors[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    // 获取输出数据指针
    auto *output_data = reinterpret_cast<float *>(output_tensors[0].sysMem.virAddr);
    // 如果模型输出是量化的int32，需要进行反量化
    if (output_tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S32) {
        auto *int_output = reinterpret_cast<int32_t *>(output_tensors[0].sysMem.virAddr);
        auto *output_scale = reinterpret_cast<float *>(output_tensors[0].properties.scale.scaleData);
        // 反量化处理
        std::vector<float> dequantized_output(CLASSES_NUM);
        for (int i = 0; i < CLASSES_NUM; ++i) {
            dequantized_output[i] = int_output[i] * output_scale[i];
        }
        // 应用Softmax转换为概率分布
        std::vector<float> probabilities(CLASSES_NUM);
        softmax(dequantized_output.data(), probabilities.data(), CLASSES_NUM);
        // 获取Top-K结果
        std::vector<std::pair<float, int>> top_k_results;
        for (int i = 0; i < CLASSES_NUM; ++i) {
            top_k_results.push_back({probabilities[i], i});
        }
        // 排序并获取Top-K
        std::partial_sort(top_k_results.begin(), top_k_results.begin() + TOP_K, top_k_results.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first > b.first;
                          });
        std::cout << "✓ Top " << TOP_K << " classification results:" << std::endl;
        for (int i = 0; i < TOP_K; ++i) {
            std::cout << "Class " << top_k_results[i].second << ": " << std::fixed << std::setprecision(4) << top_k_results[i].first << std::endl;
        }
        // 打印最终分类结果
        std::cout << "\033[32m-> Final Classification Results\033[0m" << std::endl;
        for (int i = 0; i < TOP_K; ++i) {
            int class_id = top_k_results[i].second;
            float confidence = top_k_results[i].first;
            std::string class_name = (static_cast<size_t>(class_id) < imagenet_classes.size()) ? 
                                     imagenet_classes[class_id] : "Unknown";
            std::cout << "Rank " << (i + 1) << ": " << class_name 
                      << " (Class " << class_id << ")" 
                      << " - Confidence: " << std::fixed << std::setprecision(4) << confidence << std::endl;
        }
            } else {
        // 如果输出是float类型，直接处理
        std::vector<float> raw_output(CLASSES_NUM);
        for (int i = 0; i < CLASSES_NUM; ++i) {
            raw_output[i] = output_data[i];
        }
        // 应用Softmax转换为概率分布
        std::vector<float> probabilities(CLASSES_NUM);
        softmax(raw_output.data(), probabilities.data(), CLASSES_NUM);
        // 获取Top-K结果
        std::vector<std::pair<float, int>> top_k_results;
        for (int i = 0; i < CLASSES_NUM; ++i) {
            top_k_results.push_back({probabilities[i], i});
        }
        // 排序并获取Top-K
        std::partial_sort(top_k_results.begin(), top_k_results.begin() + TOP_K, top_k_results.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first > b.first;
                          });
        std::cout << "✓ Top " << TOP_K << " classification results:" << std::endl;
        for (int i = 0; i < TOP_K; ++i) {
            std::cout << "Class " << top_k_results[i].second << ": " << std::fixed << std::setprecision(4) << top_k_results[i].first << std::endl;
        }
        // 打印最终分类结果
        std::cout << "\033[32m-> Final Classification Results\033[0m" << std::endl;
        for (int i = 0; i < TOP_K; ++i) {
            int class_id = top_k_results[i].second;
            float confidence = top_k_results[i].first;
            std::string class_name = (static_cast<size_t>(class_id) < imagenet_classes.size()) ? 
                                     imagenet_classes[class_id] : "Unknown";        
            std::cout << "Rank " << (i + 1) << ": " << class_name 
                      << " (Class " << class_id << ")" 
                      << " - Confidence: " << std::fixed << std::setprecision(4) << confidence << std::endl;
        }
    }
    std::cout << "\033[31m Post Process time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;
    // Step 9: 资源释放
    // Step 9: Release resources
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