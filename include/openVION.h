//
// Created by 楼胜峰 on 2024/10/28.
//

#ifndef COVYOLO_OPENVION_H
#define COVYOLO_OPENVION_H

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include "tinyxml2.h"

namespace openVION_YOLO {

    struct Detection {
        short class_id;
        float confidence;
        cv::Rect box;
    };

    class Inference {
    public:
        Inference() {}
        // Constructor to initialize the model with default input shape
        Inference(const std::string &model_path, const float &model_confidence_threshold, const float &model_NMS_threshold);
        // Constructor to initialize the model with specified input shape
        Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold);

        void SaveDetectionsAsVOCXML(const std::string &output_dir, const std::string &filename, const cv::Size &image_size, const std::vector<Detection> &detections);

        void RunInference(cv::Mat &frame);

    private:
        void InitializeModel(const std::string &model_path);
        void Preprocessing(const cv::Mat &frame);
        void PostProcessing(cv::Mat &frame);
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
