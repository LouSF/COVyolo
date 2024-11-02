//
// Created by 楼胜峰 on 2024/11/1.
//

#ifndef COVYOLO_THREADPOOL_H
#define COVYOLO_THREADPOOL_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <filesystem>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

class ThreadPool {
public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();

    void enqueue(std::function<void()> task);

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

void process_image(const std::filesystem::path& model_path, cv::Mat image,
                   const std::filesystem::path& input_folder, const std::filesystem::path& output_folder_xml, const std::filesystem::path& output_folder_img,
                   const std::filesystem::path& output_file_name,
                   const bool& is_debug, float confidence_threshold, float NMS_threshold);


#endif //COVYOLO_THREADPOOL_H
