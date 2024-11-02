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

std::string ChangeFileExtension(const std::string& filename, const std::string& new_extension) {
    namespace fs = std::filesystem;
    fs::path file_path(filename);
    file_path.replace_extension(new_extension);
    return file_path.string();
}


// 处理单张图片的函数
void process_image(const std::filesystem::path& model_path, cv::Mat image,
                   const std::filesystem::path& input_folder, const std::filesystem::path& output_folder, const std::filesystem::path& output_file_name,
                   const bool& is_debug, float confidence_threshold, float NMS_threshold) {

    if (image.empty()) {
        std::cerr << "ERROR: Could not load image " << output_file_name << std::endl;
        return;
    }

    openVION_YOLO::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

    auto start = std::chrono::high_resolution_clock::now();

    auto detections = inference.RunInference(image, is_debug);

    const auto xml_name = ChangeFileExtension(output_file_name, ".xml");
    inference.SaveDetectionsAsVOCXML(detections, output_folder, xml_name,
                                     input_folder, output_file_name);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Mark image" << output_file_name << " in " << duration.count() << " seconds" << std::endl;


    if (is_debug) {
        std::cout << "Save marked image " << output_file_name << " in " << output_folder << std::endl;
        auto output_path = output_folder / output_file_name;
        cv::imwrite(output_path, image);
    }


}