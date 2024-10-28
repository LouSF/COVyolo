//
// Created by 楼胜峰 on 2024/10/28.
//

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

void print_usage() {
    std::cout << "Usage: -dir <directory> -model <model> -out <output_file>\n";
}

int main(int argc, char *argv[]) {
    try {
        if (argc != 7) {
            print_usage();
            throw std::invalid_argument("Incorrect number of arguments.");
        }

        std::string dir;
        std::string model;
        std::string output_file;

        for (int i = 1; i < argc; i += 2) {
            std::string arg = argv[i];
            if (arg == "-dir") {
                dir = argv[i + 1];
            } else if (arg == "-model") {
                model = argv[i + 1];
            } else if (arg == "-out") {
                output_file = argv[i + 1];
            } else {
                print_usage();
                throw std::invalid_argument("Unknown argument: " + arg);
            }
        }

        if (dir.empty() || model.empty() || output_file.empty()) {
            print_usage();
            throw std::invalid_argument("Missing required arguments.");
        }

        std::ofstream ofs(output_file);
        if (!ofs) {
            throw std::runtime_error("Failed to open output file: " + output_file);
        }

        ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        ofs << "<configuration>\n";
        ofs << "  <directory>" << dir << "</directory>\n";
        ofs << "  <model>" << model << "</model>\n";
        ofs << "</configuration>\n";

        ofs.close();
        std::cout << "XML file created successfully: " << output_file << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
