//
// Created by 楼胜峰 on 2024/10/28.
//

#include <iostream>
#include <string>
#include <stdexcept>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <thread>
#include <queue>
#include <mutex>



#include "ThreadPool.h"


void print_usage() {
    std::cout << "Usage: ./lsf's bin file \n"
                 "--dir <directory> \n"
                 "--model <model>(option) \n"
                 "--out <output_file> \n"
                 "Debug Mode: \n"
                 "--debug(option)(output marked image) \n"
                 "--debug_IoU <IoU_NMS>(option) \n"
                 "--debug_Cof <Confidence_NMS>(option) \n" << std::endl;
}

int main(int argc, char *argv[]) {
    try {

        auto start = std::chrono::high_resolution_clock::now();

        if (argc < 5) {
            print_usage();
            throw std::invalid_argument("Incorrect number of arguments.");
        }

        std::string dir;
        std::string model;
        std::string output_folder;
        bool is_debug = false;
        float debug_IoU = 0.65;
        float debug_Cof = 0.25;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dir" && i + 1 < argc) {
                dir = argv[++i];
            } else if (arg == "--model" && i + 1 < argc) {
                model = argv[++i];
            } else if (arg == "--out" && i + 1 < argc) {
                output_folder = argv[++i];
            } else if (arg == "--debug") {
                is_debug = true;
            } else if (arg == "--debug_IoU" && i + 1 < argc) {
                debug_IoU = std::stod(argv[++i]);
            } else if (arg == "--debug_Cof" && i + 1 < argc) {
                debug_Cof = std::stod(argv[++i]);
            } else {
                print_usage();
                throw std::invalid_argument("Unknown or incomplete argument: " + arg);
            }
        }

        if (!std::filesystem::exists(output_folder)) {
            std::filesystem::create_directories(output_folder);
        }

        size_t numThreads = std::thread::hardware_concurrency();
        ThreadPool pool(numThreads);

        std::queue<std::pair<std::filesystem::path, cv::Mat>> imageQueue;
        std::mutex queueMutex;
        std::condition_variable queueCondition;
        bool doneReading = false;


        // Producer thread
        std::thread producer([&] {
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                const auto& path = entry.path();

                if (path.extension() == ".jpg" || path.extension() == ".png" || path.extension() == ".jpeg") {
                    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
                    {
                        std::lock_guard<std::mutex> lock(queueMutex);
                        imageQueue.emplace(path, image);
                    }
                    queueCondition.notify_one();
                }
            }
            doneReading = true;
            queueCondition.notify_all();
        });

        // Consumer threads
        for (size_t i = 0; i < numThreads; ++i) {
            pool.enqueue([&] {
                while (true) {
                    std::pair<std::filesystem::path, cv::Mat> item;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        queueCondition.wait(lock, [&] { return doneReading || !imageQueue.empty(); });

                        if (doneReading && imageQueue.empty())
                            return;

                        item = std::move(imageQueue.front());
                        imageQueue.pop();
                    }

                    const std::string output_path = output_folder + "/" + item.first.filename().string();
                    process_image(model, item.second, output_path, is_debug, debug_Cof, debug_IoU);
                }
            });
        }

        producer.join();

        pool.~ThreadPool();

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;

        std::cout << "All Processed in " << duration.count() << " seconds" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}