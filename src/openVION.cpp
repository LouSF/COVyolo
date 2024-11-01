//
// Created by 楼胜峰 on 2024/10/28.
//

#include "openVION.h"

#include <memory>
#include <opencv2/dnn.hpp>
#include <random>

namespace openVION_YOLO {

// Constructor to initialize the model with default input shape
    Inference::Inference(const std::string &model_path, const float &model_confidence_threshold, const float &model_NMS_threshold) {
        model_input_shape_ = cv::Size(640, 640); // Set the default size for models with dynamic shapes to prevent errors.
        model_confidence_threshold_ = model_confidence_threshold;
        model_NMS_threshold_ = model_NMS_threshold;
        InitializeModel(model_path);
    }

// Constructor to initialize the model with specified input shape
    Inference::Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold) {
        model_input_shape_ = model_input_shape;
        model_confidence_threshold_ = model_confidence_threshold;
        model_NMS_threshold_ = model_NMS_threshold;
        InitializeModel(model_path);
    }

    void Inference::InitializeModel(const std::string &model_path) {
        ov::Core core; // OpenVINO core object
        std::shared_ptr<ov::Model> model = core.read_model(model_path); // Read the model from file

        // If the model has dynamic shapes, reshape it to the specified input shape
        if (model->is_dynamic()) {
            model->reshape({1, 3, static_cast<long int>(model_input_shape_.height), static_cast<long int>(model_input_shape_.width)});
        }

        // Preprocessing setup for the model
        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
        ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255, 255, 255});
        ppp.input().model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32);
        model = ppp.build(); // Build the preprocessed model

        // Compile the model for inference
        compiled_model_ = core.compile_model(model, "AUTO");
        inference_request_ = compiled_model_.create_infer_request(); // Create inference request

        short width, height;

        // Get input shape from the model
        const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
        const ov::Shape input_shape = inputs[0].get_shape();
        height = input_shape[1];
        width = input_shape[2];
        model_input_shape_ = cv::Size2f(width, height);

        // Get output shape from the model
        const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
        const ov::Shape output_shape = outputs[0].get_shape();
        height = output_shape[1];
        width = output_shape[2];
        model_output_shape_ = cv::Size(width, height);
    }

// Method to run inference on an input frame
    void Inference::RunInference(cv::Mat &frame, const bool& is_debug) {
        Preprocessing(frame); // Preprocess the input frame
        inference_request_.infer(); // Run inference
        PostProcessing(frame, is_debug); // Postprocess the inference results
    }

// Method to preprocess the input frame
    void Inference::Preprocessing(const cv::Mat &frame) {
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, model_input_shape_, 0, 0, cv::INTER_AREA); // Resize the frame to match the model input shape

        // Calculate scaling factor
        scale_factor_.x = static_cast<float>(frame.cols / model_input_shape_.width);
        scale_factor_.y = static_cast<float>(frame.rows / model_input_shape_.height);

        float *input_data = (float *)resized_frame.data; // Get pointer to resized frame data
        const ov::Tensor input_tensor = ov::Tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), input_data); // Create input tensor
        inference_request_.set_input_tensor(input_tensor); // Set input tensor for inference
    }

// Method to postprocess the inference results
    void Inference::PostProcessing(cv::Mat &frame, const bool& is_debug) {
        std::vector<int> class_list;
        std::vector<float> confidence_list;
        std::vector<cv::Rect> box_list;

        // Get the output tensor from the inference request
        const float *detections = inference_request_.get_output_tensor().data<const float>();
        const cv::Mat detection_outputs(model_output_shape_, CV_32F, (float *)detections); // Create OpenCV matrix from output tensor

        // Iterate over detections and collect class IDs, confidence scores, and bounding boxes
        for (int i = 0; i < detection_outputs.cols; ++i) {
            const cv::Mat classes_scores = detection_outputs.col(i).rowRange(4, detection_outputs.rows);

            cv::Point class_id;
            double score;
            cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id); // Find the class with the highest score

            // Check if the detection meets the confidence threshold
            if (score > model_confidence_threshold_) {
                class_list.push_back(class_id.y);
                confidence_list.push_back(score);

                const float x = detection_outputs.at<float>(0, i);
                const float y = detection_outputs.at<float>(1, i);
                const float w = detection_outputs.at<float>(2, i);
                const float h = detection_outputs.at<float>(3, i);

                cv::Rect box;
                box.x = static_cast<int>(x);
                box.y = static_cast<int>(y);
                box.width = static_cast<int>(w);
                box.height = static_cast<int>(h);
                box_list.push_back(box);
            }
        }

        // Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
        std::vector<int> NMS_result;
        cv::dnn::NMSBoxes(box_list, confidence_list, model_confidence_threshold_, model_NMS_threshold_, NMS_result);

        // Collect final detections after NMS
        for (int i = 0; i < NMS_result.size(); ++i) {
            Detection result;
            const unsigned short id = NMS_result[i];

            result.class_id = class_list[id];
            result.confidence = confidence_list[id];
            result.box = GetBoundingBox(box_list[id]);

            if(is_debug)
                DrawDetectedObject(frame, result);
        }
    }

// Method to get the bounding box in the correct scale
    cv::Rect Inference::GetBoundingBox(const cv::Rect &src) const {
        cv::Rect box = src;
        box.x = (box.x - box.width / 2) * scale_factor_.x;
        box.y = (box.y - box.height / 2) * scale_factor_.y;
        box.width *= scale_factor_.x;
        box.height *= scale_factor_.y;
        return box;
    }

    void Inference::DrawDetectedObject(cv::Mat &frame, const Detection &detection) const {
        const cv::Rect &box = detection.box;
        const float &confidence = detection.confidence;
        const int &class_id = detection.class_id;

        // Generate a random color for the bounding box
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(120, 255);
        const cv::Scalar &color = cv::Scalar(dis(gen), dis(gen), dis(gen));

        // Draw the bounding box around the detected object
        cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), color, 3);

        // Prepare the class label and confidence text
        std::string classString = classes_[class_id] + std::to_string(confidence).substr(0, 4);

        // Get the size of the text box
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        // Draw the text box
        cv::rectangle(frame, textBox, color, cv::FILLED);

        // Put the class label and confidence text above the bounding box
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 2, 0);
    }

    void Inference::SaveDetectionsAsVOCXML(const std::vector<int>& NMS_result, const std::vector<int>& class_list,
                                           const std::vector<float>& confidence_list, const std::vector<cv::Rect>& box_list,
                                           cv::Mat& frame) {
        using namespace tinyxml2;
        using namespace std::filesystem;

        // Create XMLDocument as a local object
        XMLDocument doc;

        // Create XML declaration
        XMLDeclaration *decl = doc.NewDeclaration("xml version=\"1.0\" encoding=\"UTF-8\"");
        doc.LinkEndChild(decl);

        // Create root element
        XMLElement *annotation = doc.NewElement("annotation");
        doc.LinkEndChild(annotation);

        // Add folder and filename elements
        XMLElement* folder = doc.NewElement("folder");
        folder->SetText("images");
        annotation->InsertEndChild(folder);

        XMLElement* filename = doc.NewElement("filename");
        filename->SetText("image.jpg");
        annotation->InsertEndChild(filename);

        // Add source element
        XMLElement* source = doc.NewElement("source");
        annotation->InsertEndChild(source);

        XMLElement* database = doc.NewElement("database");
        database->SetText("Unknown");
        source->InsertEndChild(database);

        // Add size element
        XMLElement* size = doc.NewElement("size");
        annotation->InsertEndChild(size);

        XMLElement* width = doc.NewElement("width");
        width->SetText(frame.cols);
        size->InsertEndChild(width);

        XMLElement* height = doc.NewElement("height");
        height->SetText(frame.rows);
        size->InsertEndChild(height);

        tinyxml2::XMLElement* depth = doc.NewElement("depth");
        depth->SetText(frame.channels());
        size->InsertEndChild(depth);

        // Add segmented element
        tinyxml2::XMLElement* segmented = doc.NewElement("segmented");
        segmented->SetText("0");
        annotation->InsertEndChild(segmented);

        for (int i = 0; i < NMS_result.size(); ++i) {
            Detection result;
            const unsigned short id = NMS_result[i];

            result.class_id = class_list[id];
            result.confidence = confidence_list[id];
            result.box = GetBoundingBox(box_list[id]);

            XMLElement* object = doc.NewElement("object");
            annotation->InsertEndChild(object);

            XMLElement* name = doc.NewElement("name");
            name->SetText(classes_[result.class_id].c_str());
            object->InsertEndChild(name);

            XMLElement* pose = doc.NewElement("pose");
            pose->SetText("Unspecified");
            object->InsertEndChild(pose);

            XMLElement* truncated = doc.NewElement("truncated");
            truncated->SetText("0");
            object->InsertEndChild(truncated);

            XMLElement* difficult = doc.NewElement("difficult");
            difficult->SetText("0");
            object->InsertEndChild(difficult);

            XMLElement* confidence = doc.NewElement("confidence");
            confidence->SetText(result.confidence);
            object->InsertEndChild(confidence);

            XMLElement* bndbox = doc.NewElement("bndbox");
            object->InsertEndChild(bndbox);

            XMLElement* xmin = doc.NewElement("xmin");
            xmin->SetText(result.box.x);
            bndbox->InsertEndChild(xmin);

            XMLElement* ymin = doc.NewElement("ymin");
            ymin->SetText(result.box.y);
            bndbox->InsertEndChild(ymin);

            XMLElement* xmax = doc.NewElement("xmax");
            xmax->SetText(result.box.x + result.box.width);
            bndbox->InsertEndChild(xmax);

            XMLElement* ymax = doc.NewElement("ymax");
            ymax->SetText(result.box.y + result.box.height);
            bndbox->InsertEndChild(ymax);
        }
//
//        // Ensure the output directory exists
//        create_directories(output_dir);
//
//        // Construct the full file path
//        path file_path = path(output_dir) / filename;
//
//        // Save the XML to file
//        doc.SaveFile(file_path.string().c_str());
    }


} // namespace yolo
