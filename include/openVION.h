//
// Created by 楼胜峰 on 2024/10/28.
//

#ifndef COVYOLO_OPENVION_H
#define COVYOLO_OPENVION_H

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <filesystem>

#include "tinyxml2.h"

namespace openVION_YOLO {

    struct Detection {
        short class_id;
        float confidence;
        cv::Rect box;
    };

    struct Image_Detection_xml {
        std::vector<Detection> detection;
        int image_size_x, image_size_y;
    };

    class Inference {
    public:
        Inference() {}
        // Constructor to initialize the model with default input shape
        Inference(const std::string &model_path, const float &model_confidence_threshold, const float &model_NMS_threshold);
        // Constructor to initialize the model with specified input shape
        Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold);

        void SaveDetectionsAsVOCXML(const Image_Detection_xml& detections_xml, const std::filesystem::path& xml_path, const std::filesystem::path& xml_file_name,
                                    const std::filesystem::path& img_file_path, const std::filesystem::path& img_file_name) const;

        Image_Detection_xml RunInference(cv::Mat &frame ,const bool& is_debug);

    private:
        void InitializeModel(const std::string &model_path);
        void Preprocessing(const cv::Mat &frame);
        Image_Detection_xml PostProcessing(cv::Mat &frame, const bool& is_debug);
        cv::Rect GetBoundingBox(const cv::Rect &src) const;
        void DrawDetectedObject(cv::Mat &frame, const Detection &detections) const;

        cv::Point2f scale_factor_;			// Scaling factor for the input frame
        cv::Size2f model_input_shape_;	// Input shape of the model
        cv::Size model_output_shape_;		// Output shape of the model

        ov::InferRequest inference_request_;  // OpenVINO inference request
        ov::CompiledModel compiled_model_;    // OpenVINO compiled model

        float model_confidence_threshold_;  // Confidence threshold for detections
        float model_NMS_threshold_;         // Non-Maximum Suppression threshold

        std::vector<std::string> classes_ {"Header", "Text", "Figure", "Title", "Foot",};
    };

} // namespace yolo

#endif //COVYOLO_OPENVION_H
