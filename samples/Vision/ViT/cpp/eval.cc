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
#define MODEL_PATH "rdk_model_zoo_s/samples/Vision/ViT/source/reference_hbm_models/vit_cifar10_batch1_int8.hbm"

// 测试图片文件夹路径
// Path of the test images folder
#define TEST_IMGS_FOLDER "rdk_model_zoo_s/samples/Vision/ViT/source/imgs/eval_img"

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

// 测试样本数量 (每个类别100张图片，总共1000张)
// Number of test samples (100 images per class, 1000 total)
#define TEST_SAMPLES_PER_CLASS 100
#define TOTAL_TEST_SAMPLES 1000

// C/C++ Standard Libraries
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <string>

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

// 测试图片信息结构
struct TestImageInfo {
    std::string file_path;
    int true_label;
    std::string class_name;
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

// 从文件名解析真实标签
int parse_true_label_from_filename(const std::string& filename) {
    for (int i = 0; i < (int)imagenet_classes.size(); i++) {
        if (filename.find(imagenet_classes[i]) == 0) {
            return i;
        }
    }
    return -1;  // 未找到匹配的类别
}

// 获取文件夹中的所有图片文件
std::vector<TestImageInfo> get_test_images(const std::string& folder_path) {
    std::vector<TestImageInfo> test_images;
    
    // 遍历每个类别文件夹
    for (int class_id = 0; class_id < (int)imagenet_classes.size(); class_id++) {
        std::string class_name = imagenet_classes[class_id];
        std::string class_folder = folder_path + "/" + class_name;
        
        // 检查类别文件夹是否存在
        if (!std::filesystem::exists(class_folder)) {
            std::cout << "[WARNING] Class folder not found: " << class_folder << std::endl;
            continue;
        }
        
        // 遍历类别文件夹中的图片文件
        for (const auto& entry : std::filesystem::directory_iterator(class_folder)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string extension = entry.path().extension().string();
                
                // 只处理图片文件
                if (extension == ".png" || extension == ".jpg" || extension == ".jpeg") {
                    TestImageInfo img_info;
                    img_info.file_path = entry.path().string();
                    img_info.true_label = class_id;  // 文件夹名就是类别ID
                    img_info.class_name = class_name;
                    test_images.push_back(img_info);
                }
            }
        }
    }
    
    // 按类别和文件名排序，确保处理顺序一致
    std::sort(test_images.begin(), test_images.end(), 
              [](const TestImageInfo& a, const TestImageInfo& b) {
                  if (a.true_label != b.true_label) {
                      return a.true_label < b.true_label;
                  }
                  return a.file_path < b.file_path;
              });
    
    return test_images;
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
    std::cout << "[INFO] TEST_IMGS_FOLDER: " << TEST_IMGS_FOLDER << std::endl;
    std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
    std::cout << "[INFO] TOP_K: " << TOP_K << std::endl;
    std::cout << "[INFO] INPUT_HEIGHT: " << INPUT_HEIGHT << std::endl;
    std::cout << "[INFO] INPUT_WIDTH: " << INPUT_WIDTH << std::endl;
    std::cout << "[INFO] TEST_SAMPLES_PER_CLASS: " << TEST_SAMPLES_PER_CLASS << std::endl;
    std::cout << "[INFO] TOTAL_TEST_SAMPLES: " << TOTAL_TEST_SAMPLES << std::endl;

    // Step 2: 获取测试图片列表
    // Step 2: Get test images list
    std::cout << "\033[32m-> Loading test images\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now();
    
    std::vector<TestImageInfo> test_image_list = get_test_images(TEST_IMGS_FOLDER);
    if (test_image_list.empty()) {
        std::cout << "[ERROR] No test images found in folder: " << TEST_IMGS_FOLDER << std::endl;
        return -1;
    }
    
    std::cout << "✓ Loaded " << test_image_list.size() << " test images" << std::endl;
    std::cout << "\033[31m Load test images time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;

    // Step 3: 获取模型句柄
    // Step 3: Get model handle
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

    // Step 4: 检查模型输入
    // Step 4: Check model input
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

    // Step 5: 检查模型输出 - S100 ViT 按照Readme导出后应该有1个输出
    // Step 5: Check model output - S100 ViT should have 1 output according to Readme
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

    // Step 6: 准备输入输出tensor
    // Step 6: Prepare input and output tensors
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

    // Step 7: 开始批量精度评估
    // Step 7: Start batch accuracy evaluation
    std::cout << "\033[32m-> Starting batch accuracy evaluation\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now();
    
    int correct_top1 = 0;
    int correct_top5 = 0;
    // 设置UCP调度参数
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;  // 使用任意BPU核心
    
    // 评估每个测试样本
    for (int sample_idx = 0; sample_idx < (int)test_image_list.size(); sample_idx++) {
        // 获取当前样本的真实标签
        int true_label = test_image_list[sample_idx].true_label;
        
        // Step 7.1: 前处理 - 图像预处理
        // Step 7.1: Preprocessing - Image preprocessing
        
        cv::Mat img = cv::imread(test_image_list[sample_idx].file_path);
        if (img.empty()) {
            std::cout << "[ERROR] Failed to load image: " << test_image_list[sample_idx].file_path << std::endl;
            continue;
        }
        
        // 前处理参数
        float y_scale = 1.0, x_scale = 1.0;
        int x_shift = 0, y_shift = 0;
        cv::Mat resize_img;
        
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
            (void)x_scale; (void)y_scale; 
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
            (void)x_scale; (void)y_scale;
        }

        // Step 7.2: 前处理 - 颜色空间转换
        // Step 7.2: Preprocessing - Color space conversion
        
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

        // Step 7.3: 前处理 - 数据复制到tensor
        // Step 7.3: Preprocessing - Copy data to tensors
        
        // 复制数据到输入tensor
        for (int i = 0; i < input_count; i++) {
            if (i == 0) {
                // 第一个输入：复制Y分量
                memcpy(input_tensors[i].sysMem.virAddr, ynv12, INPUT_HEIGHT * INPUT_WIDTH);
            } else {
                // 第二个输入：复制UV分量 
                uint8_t *uv_src = ynv12 + INPUT_HEIGHT * INPUT_WIDTH;  // UV数据在Y之后
                int data_size = (INPUT_HEIGHT / 2) * (INPUT_WIDTH / 2) * 2;
                memcpy(input_tensors[i].sysMem.virAddr, uv_src, data_size);
            }
            
            // 刷新内存缓存，确保数据写入到物理内存
            hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
        }

        // Step 7.4: 推理执行
        // Step 7.4: Inference execution
        
        // 生成推理任务句柄
        hbUCPTaskHandle_t task_handle = nullptr;
        int infer_ret = hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle);
        if (infer_ret != 0) {
            std::cout << "[ERROR] hbDNNInferV2 failed with error code: " << infer_ret << std::endl;
            continue;
        }
        
        // 提交任务到UCP调度器
        int submit_ret = hbUCPSubmitTask(task_handle, &ctrl_param);
        if (submit_ret != 0) {
            std::cout << "[ERROR] hbUCPSubmitTask failed with error code: " << submit_ret << std::endl;
            hbUCPReleaseTask(task_handle);
            continue;
        }
        
        // 等待任务完成，设置10秒超时
        int wait_ret = hbUCPWaitTaskDone(task_handle, 10000);
        if (wait_ret != 0) {
            std::cout << "[ERROR] hbUCPWaitTaskDone failed with error code: " << wait_ret << std::endl;
            hbUCPReleaseTask(task_handle);
            continue;
        }
        
        // 释放任务句柄
        hbUCPReleaseTask(task_handle);

        // Step 7.5: 后处理 - 获取分类结果
        // Step 7.5: Post-processing - Get classification results
        
        // 刷新BPU内存，确保获取到最新的推理结果
        hbUCPMemFlush(&(output_tensors[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
        
        // 检查输出数据指针是否有效
        if (output_tensors[0].sysMem.virAddr == nullptr) {
            std::cout << "[ERROR] Output tensor virtual address is null" << std::endl;
            continue;
        }
        
        // 获取输出数据指针
        auto *output_data = reinterpret_cast<float *>(output_tensors[0].sysMem.virAddr);
        
        // 检查输出数据是否有效
        if (output_data == nullptr) {
            std::cout << "[ERROR] Output data pointer is null" << std::endl;
            continue;
        }
        
        // 处理输出数据 - 检查是否为量化模型
        std::vector<float> scores(CLASSES_NUM);
        
        if (output_tensors[0].properties.scale.scaleData != nullptr) {
            // 量化模型 - 需要反量化
            auto *output_scale = reinterpret_cast<float *>(output_tensors[0].properties.scale.scaleData);
            for (int i = 0; i < CLASSES_NUM; i++) {
                scores[i] = output_data[i] * output_scale[0];
            }
        } else {
            // 非量化模型 - 直接使用输出数据
            for (int i = 0; i < CLASSES_NUM; i++) {
                scores[i] = output_data[i];
            }
        }
        
        // 应用Softmax获取概率分布
        std::vector<float> probabilities(CLASSES_NUM);
        softmax(scores.data(), probabilities.data(), CLASSES_NUM);
        
        // 获取Top-K结果
        std::vector<std::pair<float, int>> score_index_pairs;
        for (int i = 0; i < CLASSES_NUM; i++) {
            score_index_pairs.push_back({probabilities[i], i});
        }
        
        // 按概率降序排序
        std::sort(score_index_pairs.begin(), score_index_pairs.end(), 
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                      return a.first > b.first;
                  });
        
        // Step 7.6: 精度统计
        // Step 7.6: Accuracy statistics
        
        // 检查Top-1准确率 
        if (score_index_pairs[0].second == true_label) {
            correct_top1++;
        }
        
        // 检查Top-5准确率
        for (int i = 0; i < std::min(5, (int)score_index_pairs.size()); i++) {
            if (score_index_pairs[i].second == true_label) {
                correct_top5++;
                break;
            }
        }
        
        // 每100个样本打印一次进度和调试信息
        if (sample_idx % 100 == 0) {
            std::cout << "Processing sample " << sample_idx << "/" << test_image_list.size() << std::endl;
        }
    }
    
    std::cout << "\033[31m Total evaluation time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;

    // Step 8: 计算并打印精度结果
    // Step 8: Calculate and print accuracy results
    std::cout << "\033[32m-> Accuracy Evaluation Results\033[0m" << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│                    Accuracy Results                     │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    
    float top1_accuracy = (float)correct_top1 / test_image_list.size() * 100.0f;
    float top5_accuracy = (float)correct_top5 / test_image_list.size() * 100.0f;
    
    std::cout << "│ Total Test Samples: " << std::setw(37) << test_image_list.size() << " │" << std::endl;
    std::cout << "│ Correct Top-1 Predictions: " << std::setw(30) << correct_top1 << " │" << std::endl;
    std::cout << "│ Correct Top-5 Predictions: " << std::setw(30) << correct_top5 << " │" << std::endl;
    std::cout << "│ Top-1 Accuracy: " << std::setw(38) << std::fixed << std::setprecision(2) << top1_accuracy << "% │" << std::endl;
    std::cout << "│ Top-5 Accuracy: " << std::setw(38) << std::fixed << std::setprecision(2) << top5_accuracy << "% │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────┘" << std::endl;
    
    // 打印详细信息
    std::cout << "\n\033[32m-> Detailed Results\033[0m" << std::endl;
    std::cout << "Top-1 Accuracy: " << std::fixed << std::setprecision(4) << top1_accuracy << "% (" 
              << correct_top1 << "/" << test_image_list.size() << ")" << std::endl;
    std::cout << "Top-5 Accuracy: " << std::fixed << std::setprecision(4) << top5_accuracy << "% (" 
              << correct_top5 << "/" << test_image_list.size() << ")" << std::endl;

    // Step 9: 资源释放
    // Step 9: Release resources
    std::cout << "\033[32m-> Cleaning up resources\033[0m" << std::endl;
    
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
    
    std::cout << "\033[32m✓ Accuracy evaluation completed successfully\033[0m" << std::endl;
    return 0;
}