//
// Created by 楼胜峰 on 2024/11/1.
//

#include "ThreadPool.h"
#include "openVION.h"

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });

                    if (this->stop && this->tasks.empty())
                        return;

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
        worker.join();
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.emplace(task);
    }
    condition.notify_one();
}

// 处理单张图片的函数
void process_image(const std::string& model_path, cv::Mat image, const std::string& output_path,
                   const bool& is_debug, float confidence_threshold, float NMS_threshold) {

    if (image.empty()) {
        std::cerr << "ERROR: Could not load image " << output_path << std::endl;
        return;
    }

    openVION_YOLO::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

    auto start = std::chrono::high_resolution_clock::now();

    inference.RunInference(image, is_debug);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Processed " << output_path << " in " << duration.count() << " seconds" << std::endl;

    if (is_debug)
        cv::imwrite(output_path, image);

}