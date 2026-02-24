cmake_minimum_required(VERSION 3.10)

project(ultralytics_yolo_pose_nashe)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译器标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 设置OpenCV包
find_package(OpenCV REQUIRED)

# S100 UCP库路径配置
set(HOBOT_INCLUDE_PATH "/usr/include")
set(HOBOT_LIB_PATH "/usr/hobot/lib")

# 包含头文件
include_directories(${HOBOT_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})

# 链接库路径
link_directories(${HOBOT_LIB_PATH})

# 添加可执行文件
add_executable(ultralytics_yolo_pose main.cc)

# 链接库
target_link_libraries(ultralytics_yolo_pose
                      ${OpenCV_LIBS}    # OpenCV库
                      dnn               # S100 DNN推理库
                      hbucp             # S100 UCP统一计算平台库
                      pthread           # 多线程支持
                      rt                # 实时库
                      dl                # 动态链接库支持
                      m                 # 数学库
                      )

# 打印调试信息
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "HOBOT_INCLUDE_PATH: ${HOBOT_INCLUDE_PATH}")
message(STATUS "HOBOT_LIB_PATH: ${HOBOT_LIB_PATH}")