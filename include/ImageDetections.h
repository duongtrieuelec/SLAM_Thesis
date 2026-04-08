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

#ifndef IMAGE_DETECTIONS_H
#define IMAGE_DETECTIONS_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#include "Utils.h"
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

class Detection {
public:
  typedef std::shared_ptr<Detection> Ptr;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Detection(unsigned int cat, double det_score, const BBox2 bb)
      : category_id(cat), score(det_score), bbox(bb) {}

  friend std::ostream &operator<<(std::ostream &os, const Detection &det);
  unsigned int category_id;
  double score;
  BBox2 bbox;

private:
  Detection() = delete;
};

class ImageDetectionsManager {
public:
  ImageDetectionsManager() {}
  virtual ~ImageDetectionsManager() {}

  virtual std::vector<Detection::Ptr> detect(const std::string &name) const = 0;
  virtual std::vector<Detection::Ptr> detect(unsigned int idx) const = 0;
  virtual std::vector<Detection::Ptr> detect(cv::Mat img) const = 0;

  // Get person segmentation mask (for segmentation-based filtering)
  // Returns empty Mat if not supported by the detector
  virtual cv::Mat getPersonMask() const { return cv::Mat(); }
};

class DetectionsFromFile : public ImageDetectionsManager {
public:
  DetectionsFromFile(const std::string &filename,
                     const std::vector<int> &cats_to_ignore);
  ~DetectionsFromFile() {}

  std::vector<Detection::Ptr> detect(const std::string &name) const;

  std::vector<Detection::Ptr> detect(unsigned int idx) const;

  std::vector<Detection::Ptr> detect(cv::Mat img) const {
    std::cerr << "This function is not available. You should pass the image "
                 "filename or index as input\n";
    return {};
  }

private:
  std::unordered_map<std::string, std::vector<Detection::Ptr>> detections_;
  std::vector<std::string> frame_names_;
  json data_;
};

#ifdef USE_DNN
class ObjectDetector : public ImageDetectionsManager {
public:
  ObjectDetector(const std::string &model,
                 const std::vector<int> &cats_to_ignore);
  ~ObjectDetector() {}

  std::vector<Detection::Ptr> detect(cv::Mat img) const;

  std::vector<Detection::Ptr> detect(const std::string &name) const {
    std::cerr << "This function is not available. You should pass an image as "
                 "input\n";
    return {};
  }
  std::vector<Detection::Ptr> detect(unsigned int idx) const {
    std::cerr << "This function is not available. You should pass an image as "
                 "input\n";
    return {};
  }

private:
  // cv::dnn::Net network_;
  std::unique_ptr<cv::dnn::Net> network_;
  std::unordered_set<int> ignored_cats_;
};

// Segmentation detector using YOLOv8-seg model
// Generates binary mask of person regions for keypoint filtering
class SegmentationDetector : public ImageDetectionsManager {
public:
  SegmentationDetector(const std::string &model,
                       const std::vector<int> &cats_to_ignore);
  ~SegmentationDetector() {}

  std::vector<Detection::Ptr> detect(cv::Mat img) const;

  // Get the combined person mask from the last detection
  cv::Mat getPersonMask() const { return person_mask_; }

  std::vector<Detection::Ptr> detect(const std::string &name) const {
    std::cerr << "This function is not available. You should pass an image as "
                 "input\n";
    return {};
  }
  std::vector<Detection::Ptr> detect(unsigned int idx) const {
    std::cerr << "This function is not available. You should pass an image as "
                 "input\n";
    return {};
  }

private:
  std::unique_ptr<cv::dnn::Net> network_;
  std::unordered_set<int> ignored_cats_;
  mutable cv::Mat person_mask_; // Combined mask of all person detections

  // Process mask prototypes and coefficients to generate instance masks
  void processSegmentation(const cv::Mat &output0, const cv::Mat &output1,
                           int img_width, int img_height, float x_factor,
                           float y_factor) const;
};
#else
class ObjectDetector : public ImageDetectionsManager {
public:
  ObjectDetector(const std::string &model,
                 const std::vector<int> &cats_to_ignore) {
    std::cerr << "Object detection is only available with dnn module of opencv "
                 ">= 4\n";
  }
  ~ObjectDetector() {}

  std::vector<Detection::Ptr> detect(cv::Mat img) const {
    std::cerr << "Object detection is only available with dnn module of opencv "
                 ">= 4\n";
    return {};
  }

  std::vector<Detection::Ptr> detect(const std::string &name) const {
    std::cerr << "Object detection is only available with dnn module of opencv "
                 ">= 4\n";
    return {};
  }
  std::vector<Detection::Ptr> detect(unsigned int idx) const {
    std::cerr << "Object detection is only available with dnn module of opencv "
                 ">= 4\n";
    return {};
  }
};
#endif

} // namespace ORB_SLAM2

#endif