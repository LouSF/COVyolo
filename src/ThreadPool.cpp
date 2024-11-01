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
void process_image(const std::string& model_path, const std::filesystem::path& image_path, const std::string& output_folder,
                   bool is_debug, float confidence_threshold, float NMS_threshold) {
    cv::Mat image = cv::imread(image_path.string());

    if (image.empty()) {
        std::cerr << "ERROR: Could not load image " << image_path << std::endl;
        return;
    }

    openVION_YOLO::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

    auto start = std::chrono::high_resolution_clock::now();
    inference.RunInference(image);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Processed " << image_path.filename() << " in " << duration.count() << " seconds" << std::endl;

    std::string output_path = output_folder + "/" + image_path.filename().string();
    cv::imwrite(output_path, image);
}

