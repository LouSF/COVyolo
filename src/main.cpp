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

#include "argparse/argparse.hpp"
#include "ThreadPool.h"

int main(int argc, char *argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    argparse::ArgumentParser YOLO_inference("lsf's OpenVNO based YOLO inference");

    YOLO_inference.add_argument("-d", "--dir")
            .required()
            .help("Input Folder");

    YOLO_inference.add_argument("-m", "--model")
            .default_value(std::string("model/model.xml"))
            .help("model (option)");

    YOLO_inference.add_argument("-o", "--out")
            .required()
            .help("Output Folder");

    YOLO_inference.add_argument("--debug")
            .default_value(std::string(""))
            .help("(option)(output marked image) <output_labeled_image_file>(option)");

    YOLO_inference.add_argument("--debug_IoU")
            .default_value(0.0f)
            .scan<'g', float>()
            .help("<IoU_NMS>(option)");

    YOLO_inference.add_argument("--debug_Cof")
            .default_value(0.0f)
            .scan<'g', float>()
            .help("<Confidence_NMS>(option)");

    try {
        YOLO_inference.parse_args(argc, argv);

        auto dir = YOLO_inference.get<std::string>("--dir");
        auto model = YOLO_inference.get<std::string>("--model");
        auto output_folder = YOLO_inference.get<std::string>("--out");
        auto debug_path = YOLO_inference.get<std::string>("--debug");
        auto debug_IoU = YOLO_inference.get<float>("--debug_IoU");
        auto debug_Cof = YOLO_inference.get<float>("--debug_Cof");

        std::filesystem::path dir_path(dir);
        std::filesystem::path model_path(model);
        std::filesystem::path output_folder_path(output_folder);
        std::filesystem::path debug_path_path(debug_path);

        bool is_debug = !debug_path_path.empty();

        if (!std::filesystem::exists(output_folder_path)) {
            std::filesystem::create_directories(output_folder_path);
        }

        size_t numThreads = std::thread::hardware_concurrency();
        ThreadPool pool(numThreads);

        std::queue<std::pair<std::filesystem::path, cv::Mat>> imageQueue;
        std::mutex queueMutex;
        std::condition_variable queueCondition;
        bool doneReading = false;

        // Producer thread
        std::thread producer([&] {
            for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
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
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                doneReading = true;
            }
            queueCondition.notify_all();
        });

        // Consumer threads
        std::vector<std::thread> consumers;
        for (size_t i = 0; i < numThreads; ++i) {
            consumers.emplace_back([&] {
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

                    process_image(model_path, item.second,
                                  dir_path, output_folder_path, debug_path_path,
                                  item.first.filename(),
                                  is_debug, debug_Cof, debug_IoU);
                }
            });
        }

        producer.join();
        for (auto& consumer : consumers) {
            consumer.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "All Processed in " << duration.count() << " seconds" << std::endl;

    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << YOLO_inference;
        std::exit(1);
    }

    return 0;
}