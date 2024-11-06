cmake_minimum_required(VERSION 3.10)

project(COVyolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_C_FLAGS_DEBUG "$ENV{CMAKE_C_FLAGS_DEBUG} -g")
#set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CMAKE_CXX_FLAGS_DEBUG} -g")

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(Threads REQUIRED)

add_library(argparse INTERFACE)
add_library(argparse::argparse ALIAS argparse)
#message(${OpenCV_INCLUDE_DIRS})
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${OpenVINO_INCLUDE_DIRS}
        lib/argparse/include
        include
        model
)

add_subdirectory(
        lib/tinyxml2
        lib/argparse
)

file(GLOB SOURCES "src/*.cpp"
)

add_executable(COVyolo
        ${SOURCES}
)

target_link_libraries(COVyolo
        ${OpenCV_LIBS}
        ${OpenVINO_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        openvino::runtime
        tinyxml2
)

set_target_properties(COVyolo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

install(DIRECTORY model/ DESTINATION ${CMAKE_SOURCE_DIR}/bin/model)
