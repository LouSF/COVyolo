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

void process_image(const std::string& model_path, const std::filesystem::path& image_path, const std::string& output_folder,
                   const bool& is_debug, float confidence_threshold, float NMS_threshold);


#endif //COVYOLO_THREADPOOL_H
