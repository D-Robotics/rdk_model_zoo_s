cmake_minimum_required(VERSION 3.10)

project(yolo26_detect)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(OpenCV REQUIRED)

set(HOBOT_INCLUDE_PATH "/usr/include")
set(HOBOT_LIB_PATH "/usr/hobot/lib")

include_directories(${HOBOT_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

link_directories(${HOBOT_LIB_PATH})

add_executable(yolo26_detect main.cc)

target_link_libraries(yolo26_detect
                      ${OpenCV_LIBS}
                      dnn
                      hbucp
                      pthread
                      rt
                      dl
                      m
                      )

message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "HOBOT_INCLUDE_PATH: ${HOBOT_INCLUDE_PATH}")
message(STATUS "HOBOT_LIB_PATH: ${HOBOT_LIB_PATH}")
