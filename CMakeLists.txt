cmake_minimum_required(VERSION 3.10)

# 项目名称
project(COVyolo)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "$ENV{CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CMAKE_CXX_FLAGS_DEBUG} -g")

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 查找 OpenVINO
find_package(OpenVINO REQUIRED)

# 包含头文件目录
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${OpenVINO_INCLUDE_DIRS}
        include
        model
)

# 添加 tinyxml2 子目录
add_subdirectory(lib/tinyxml2)

# 查找源文件
file(GLOB SOURCES "src/*.cpp")

# 添加可执行文件
add_executable(COVyolo ${SOURCES})

# 链接库
target_link_libraries(COVyolo
        ${OpenCV_LIBS}
        ${OpenVINO_LIBRARIES}
        openvino::runtime
        tinyxml2
)

# 设置输出目录
set_target_properties(COVyolo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# 添加模型目录
install(DIRECTORY model/ DESTINATION ${CMAKE_SOURCE_DIR}/bin/model)
