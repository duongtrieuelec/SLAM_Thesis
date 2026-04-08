/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
 * (Inria, LORIA, Université de Lorraine)
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "ImageDetections.h"

#include <experimental/filesystem>
#include <unordered_set>

namespace fs = std::experimental::filesystem;

namespace ORB_SLAM2 {

std::ostream &operator<<(std::ostream &os, const Detection &det) {
  os << "Detection:  cat = " << det.category_id << "  score = " << det.score
     << "  bbox = " << det.bbox.transpose();
  return os;
}

DetectionsFromFile::DetectionsFromFile(const std::string &filename,
                                       const std::vector<int> &cats_to_ignore)
    : ImageDetectionsManager() {
  std::unordered_set<unsigned int> to_ignore(cats_to_ignore.begin(),
                                             cats_to_ignore.end());
  std::ifstream fin(filename);
  if (!fin.is_open()) {
    std::cerr << "Warning failed to open file: " << filename << std::endl;
    return;
  }
  fin >> data_;

  for (auto &frame : data_) {
    std::string name = frame["file_name"].get<std::string>();
    name = fs::path(name).filename();
    frame_names_.push_back(name);

    std::vector<Detection::Ptr> detections;
    for (auto &d : frame["detections"]) {
      double score = d["detection_score"].get<double>();
      unsigned int cat = d["category_id"].get<unsigned int>();
      if (to_ignore.find(cat) != to_ignore.end())
        continue;
      auto bb = d["bbox"];
      Eigen::Vector4d bbox(bb[0], bb[1], bb[2], bb[3]);
      detections.push_back(
          std::shared_ptr<Detection>(new Detection(cat, score, bbox)));
    }
    detections_[name] = detections;
  }
}

std::vector<Detection::Ptr>
DetectionsFromFile::detect(const std::string &name) const {
  std::string basename = fs::path(name).filename();

  if (detections_.find(basename) == detections_.end()) {
    // Debug: print first few frames to help diagnose
    static int debug_count = 0;
    if (debug_count < 3) {
      std::cerr << "Debug: Looking for '" << basename
                << "' but not found in JSON." << std::endl;
      if (!frame_names_.empty()) {
        std::cerr << "  First frame in JSON: '" << frame_names_[0] << "'"
                  << std::endl;
      }
      debug_count++;
    }
    return {};
  }
  return detections_.at(basename);
}
std::vector<Detection::Ptr> DetectionsFromFile::detect(unsigned int idx) const {
  if (idx < 0 || idx >= frame_names_.size()) {
    std::cerr << "Warning invalid index: " << idx << std::endl;
    return {};
  }
  return this->detect(frame_names_[idx]);
}

#ifdef USE_DNN

ObjectDetector::ObjectDetector(const std::string &model,
                               const std::vector<int> &cats_to_ignore)
    : network_(std::make_unique<cv::dnn::Net>()),
      ignored_cats_(cats_to_ignore.begin(), cats_to_ignore.end()),
      ImageDetectionsManager() {
  if (model.substr(model.size() - 4) == "onnx")
    *network_ = cv::dnn::readNet(model);
  else
    *network_ = cv::dnn::readNetFromDarknet(model + ".cfg", model + ".weights");
  network_->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  network_->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

std::vector<Detection::Ptr> ObjectDetector::detect(cv::Mat img) const {

  // Settings
  const int INPUT_WIDTH = 640.0 / 2; // size of image passed to the network
                                     // (reducing may be faster to process)
  const int INPUT_HEIGHT = 640.0 / 2;
  const float SCORE_THRESHOLD = 0.5;
  const float NMS_THRESHOLD = 0.45;
  const float CONFIDENCE_THRESHOLD = 0.45;

  cv::Mat result;
  cv::dnn::blobFromImage(img, result, 1. / 255,
                         cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(),
                         true, false);

  std::vector<cv::Mat> predictions;

  // Get all output layer names and use forward with all outputs
  auto outnames = network_->getUnconnectedOutLayersNames();
  network_->setInput(result);
  network_->forward(predictions, outnames);
  const cv::Mat &output = predictions[0];

  float x_factor = (float)(img.cols) / INPUT_WIDTH;
  float y_factor = (float)(img.rows) / INPUT_HEIGHT;

  std::vector<int> class_ids;
  class_ids.reserve(1000);
  std::vector<float> confidences;
  confidences.reserve(1000);
  std::vector<cv::Rect> boxes;
  boxes.reserve(1000);

  // Detect YOLOv5 vs YOLOv8 based on output shape
  // YOLOv5: [1, num_detections, 5+num_classes] - e.g., [1, 25200, 85]
  // YOLOv8: [1, 4+num_classes, num_detections] - e.g., [1, 84, 8400]
  bool is_yolov8 = false;
  int rows, cols;

  if (output.dims == 3) {
    // Check if it's YOLOv8 format (dimension 1 is small, dimension 2 is large)
    if (output.size[1] < output.size[2]) {
      is_yolov8 = true;
      rows = output.size[2]; // num_detections
      cols = output.size[1]; // 4 + num_classes
    } else {
      is_yolov8 = false;
      rows = output.size[1]; // num_detections
      cols = output.size[2]; // 5 + num_classes
    }
  } else {
    rows = output.size[1];
    cols = output.size[2];
  }

  if (is_yolov8) {
    // YOLOv8 format: [1, 84, num_detections] -> need to transpose
    // 84 = 4 (x, y, w, h) + 80 (class scores)
    int nb_classes = cols - 4;

    // Create 2D matrix from 3D tensor: [1, 84, 2100] -> [84, 2100] -> transpose
    // to [2100, 84]
    cv::Mat output_2d(cols, rows, CV_32F,
                      const_cast<float *>(output.ptr<float>()));
    cv::Mat output_transposed;
    cv::transpose(output_2d, output_transposed); // [2100, 84]

    float *data = (float *)output_transposed.data;

    for (int i = 0; i < rows; ++i) {
      // Get class scores (skip first 4 values which are bbox)
      float *classes_scores = data + 4;
      cv::Mat scores(1, nb_classes, CV_32FC1, classes_scores);
      cv::Point class_id;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

      if (ignored_cats_.count(class_id.x) == 0 &&
          max_class_score > SCORE_THRESHOLD) {
        confidences.push_back(max_class_score);
        class_ids.push_back(class_id.x);

        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];
        int left = int((x - 0.5 * w) * x_factor);
        int top = int((y - 0.5 * h) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);
        boxes.push_back(cv::Rect(left, top, width, height));
      }
      data += cols;
    }
  } else {
    // YOLOv5 format: [1, num_detections, 5+num_classes]
    // output should have size: 1 x nb_detections x (5 + nb_classes)
    float *data = (float *)output.data;
    int nb_classes = cols - 5;

    for (int i = 0; i < rows; ++i) {
      float confidence = data[4];
      if (confidence >= .4) {
        float *classes_scores = data + 5;
        cv::Mat scores(1, nb_classes, CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        if (ignored_cats_.count(class_id.x) == 0 &&
            max_class_score > SCORE_THRESHOLD) {
          confidences.push_back(max_class_score);
          class_ids.push_back(class_id.x);

          float x = data[0];
          float y = data[1];
          float w = data[2];
          float h = data[3];
          int left = int((x - 0.5 * w) * x_factor);
          int top = int((y - 0.5 * h) * y_factor);
          int width = int(w * x_factor);
          int height = int(h * y_factor);
          boxes.push_back(cv::Rect(left, top, width, height));
        }
      }
      data += cols;
    }
  }

  // Filter detection with NMS
  std::vector<Detection::Ptr> detections;
  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    nms_result);
  for (int i = 0; i < nms_result.size(); i++) {
    int idx = nms_result[i];
    cv::Rect &bb = boxes[idx];
    Eigen::Vector4d bbox(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);
    detections.push_back(std::shared_ptr<Detection>(
        new Detection(class_ids[idx], confidences[idx], bbox)));
  }
  return detections;
}

// ============================================================================
// SegmentationDetector Implementation for YOLOv8-seg
// ============================================================================

SegmentationDetector::SegmentationDetector(
    const std::string &model, const std::vector<int> &cats_to_ignore)
    : network_(std::make_unique<cv::dnn::Net>()),
      ignored_cats_(cats_to_ignore.begin(), cats_to_ignore.end()),
      ImageDetectionsManager() {
  *network_ = cv::dnn::readNet(model);
  network_->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  network_->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  std::cout << "SegmentationDetector: Loaded model " << model << std::endl;
}

std::vector<Detection::Ptr> SegmentationDetector::detect(cv::Mat img) const {
  const int INPUT_WIDTH = 640;
  const int INPUT_HEIGHT = 640;
  const float SCORE_THRESHOLD = 0.5;
  const float NMS_THRESHOLD = 0.45;
  const float MASK_THRESHOLD = 0.5;

  // Preprocess
  cv::Mat blob;
  cv::dnn::blobFromImage(img, blob, 1. / 255,
                         cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(),
                         true, false);
  network_->setInput(blob);

  // Forward pass - YOLOv8-seg has 2 outputs
  std::vector<cv::Mat> outputs;
  auto outnames = network_->getUnconnectedOutLayersNames();
  network_->forward(outputs, outnames);

  // output0: (1, 116, 8400) - detections [4 bbox + 80 classes + 32 mask coeffs]
  // output1: (1, 32, 160, 160) - prototype masks
  const cv::Mat &output0 = outputs[0];
  const cv::Mat &output1 = outputs[1];

  float x_factor = (float)img.cols / INPUT_WIDTH;
  float y_factor = (float)img.rows / INPUT_HEIGHT;

  // Parse output0: YOLOv8-seg format [1, 116, 8400]
  // 116 = 4 (bbox) + 80 (classes) + 32 (mask coefficients)
  int num_detections = output0.size[2];      // 8400
  int num_channels = output0.size[1];        // 116
  int num_classes = num_channels - 4 - 32;   // 80
  int mask_coeffs_start = num_channels - 32; // 84

  // Transpose to [8400, 116]
  cv::Mat output_2d(num_channels, num_detections, CV_32F,
                    const_cast<float *>(output0.ptr<float>()));
  cv::Mat output_transposed;
  cv::transpose(output_2d, output_transposed);

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<cv::Mat>
      mask_coefficients; // Store mask coeffs for each detection

  float *data = (float *)output_transposed.data;

  for (int i = 0; i < num_detections; ++i) {
    // Get class scores
    float *classes_scores = data + 4;
    cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
    cv::Point class_id;
    double max_class_score;
    minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

    if (max_class_score > SCORE_THRESHOLD) {
      confidences.push_back(max_class_score);
      class_ids.push_back(class_id.x);

      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];
      int left = int((x - 0.5 * w) * x_factor);
      int top = int((y - 0.5 * h) * y_factor);
      int width = int(w * x_factor);
      int height = int(h * y_factor);
      boxes.push_back(cv::Rect(left, top, width, height));

      // Store mask coefficients (32 values)
      cv::Mat coeffs(1, 32, CV_32F, data + mask_coeffs_start);
      mask_coefficients.push_back(coeffs.clone());
    }
    data += num_channels;
  }

  // NMS
  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    nms_result);

  // Initialize combined person mask
  person_mask_ = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

  // Process prototype masks: (1, 32, 160, 160) -> (32, 160*160) = (32, 25600)
  int proto_h = output1.size[2];    // 160
  int proto_w = output1.size[3];    // 160
  int num_protos = output1.size[1]; // 32

  cv::Mat protos(num_protos, proto_h * proto_w, CV_32F,
                 const_cast<float *>(output1.ptr<float>()));

  std::vector<Detection::Ptr> detections;

  for (int i = 0; i < (int)nms_result.size(); i++) {
    int idx = nms_result[i];
    cv::Rect &bb = boxes[idx];

    // Clamp bounding box to image
    bb.x = std::max(0, bb.x);
    bb.y = std::max(0, bb.y);
    bb.width = std::min(bb.width, img.cols - bb.x);
    bb.height = std::min(bb.height, img.rows - bb.y);

    if (bb.width <= 0 || bb.height <= 0)
      continue;

    Eigen::Vector4d bbox(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);

    // Skip ignored categories for detections list
    if (ignored_cats_.count(class_ids[idx]) == 0) {
      detections.push_back(
          std::make_shared<Detection>(class_ids[idx], confidences[idx], bbox));
    }

    // Generate mask for person class (class_id = 0)
    if (class_ids[idx] == 0) { // Person class
      // mask = coeffs @ protos -> (1, 25600)
      cv::Mat mask_raw =
          mask_coefficients[idx] * protos; // (1, 32) x (32, 25600) = (1, 25600)

      // Reshape to (160, 160)
      cv::Mat mask_160 = mask_raw.reshape(1, proto_h);

      // Sigmoid
      cv::exp(-mask_160, mask_160);
      mask_160 = 1.0 / (1.0 + mask_160);

      // Resize to original image size
      cv::Mat mask_resized;
      cv::resize(mask_160, mask_resized, cv::Size(img.cols, img.rows), 0, 0,
                 cv::INTER_LINEAR);

      // Threshold and crop to bounding box
      cv::Mat mask_binary;
      cv::threshold(mask_resized, mask_binary, MASK_THRESHOLD, 255,
                    cv::THRESH_BINARY);
      mask_binary.convertTo(mask_binary, CV_8UC1);

      // Apply bounding box mask (crop outside bbox)
      cv::Mat bbox_mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
      cv::rectangle(bbox_mask, bb, cv::Scalar(255), cv::FILLED);
      cv::bitwise_and(mask_binary, bbox_mask, mask_binary);

      // Combine with person_mask_
      cv::bitwise_or(person_mask_, mask_binary, person_mask_);
    }
  }

  return detections;
}

#endif

} // namespace ORB_SLAM2