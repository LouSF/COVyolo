//
// Created by 楼胜峰 on 2024/10/28.
//

#include <iostream>
#include <string>
#include <stdexcept>

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
        if (argc < 5) {
            print_usage();
            throw std::invalid_argument("Incorrect number of arguments.");
        }

        std::string dir;
        std::string model;
        std::string output_file;
        bool debug = false;
        float debug_IoU = 0.65;
        float debug_Cof = 0.25;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dir" && i + 1 < argc) {
                dir = argv[++i];
            } else if (arg == "--model" && i + 1 < argc) {
                model = argv[++i];
            } else if (arg == "--out" && i + 1 < argc) {
                output_file = argv[++i];
            } else if (arg == "--debug") {
                debug = true;
            } else if (arg == "--debug_IoU" && i + 1 < argc) {
                debug_IoU = std::stod(argv[++i]);
            } else if (arg == "--debug_Cof" && i + 1 < argc) {
                debug_Cof = std::stod(argv[++i]);
            } else {
                print_usage();
                throw std::invalid_argument("Unknown or incomplete argument: " + arg);
            }
        }

        if (!std::filesystem::exists(output_file)) {
            std::filesystem::create_directories(output_file);
        }

        size_t numThreads = std::thread::hardware_concurrency();
        ThreadPool pool(numThreads);

        auto start = std::chrono::high_resolution_clock::now();

        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            const auto& path = entry.path();

            if (path.extension() == ".jpg" || path.extension() == ".png" || path.extension() == ".jpeg") {
                pool.enqueue([=] {
                    process_image(model, path, output_file, debug, debug_Cof, debug_IoU);
                });
            }
        }

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