/**
 * ORB-SLAM2 RGB-D mode
 *
 * Based on ORB-SLAM2.cc but uses RGB-D input for proper scale
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Osmap.h"
#include "Utils.h"
#include <ImageDetections.h>
#include <System.h>
#include <experimental/filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;
using namespace std;

// Load RGB-D associations file (format: rgb_timestamp rgb_file depth_timestamp
// depth_file)
void LoadAssociations(const string &strAssociationFile,
                      vector<string> &vstrImageFilenamesRGB,
                      vector<string> &vstrImageFilenamesD,
                      vector<double> &vTimestamps) {
  ifstream fAssociation;
  fAssociation.open(strAssociationFile.c_str());
  while (!fAssociation.eof()) {
    string s;
    getline(fAssociation, s);
    if (!s.empty() && s[0] != '#') {
      stringstream ss;
      ss << s;
      double t;
      string sRGB, sD;
      double tD;
      ss >> t >> sRGB >> tD >> sD;
      if (!sRGB.empty() && !sD.empty()) {
        vTimestamps.push_back(t);
        vstrImageFilenamesRGB.push_back(sRGB);
        vstrImageFilenamesD.push_back(sD);
      }
    }
  }
}

int main(int argc, char **argv) {
  srand(time(nullptr));
  std::cout << "ORB-SLAM2 RGB-D Mode" << std::endl;

  if (argc != 8) {
    cerr << endl
         << "Usage:\n"
            " ./ORB-SLAM2-rgbd\n"
            "      vocabulary_file\n"
            "      camera_file\n"
            "      path_to_dataset (folder containing associations.txt)\n"
            "      detections_file (.json file with detections or .onnx yolov5 "
            "weights)\n"
            "      categories_to_ignore_file (file containing the categories "
            "to ignore)\n"
            "      relocalization_mode ('points', 'objects' or "
            "'points+objects')\n"
            "      output_name \n";
    return 1;
  }

  std::string vocabulary_file = string(argv[1]);
  std::string parameters_file = string(argv[2]);
  string path_to_dataset = string(argv[3]);
  std::string detections_file(argv[4]);
  std::string categories_to_ignore_file(argv[5]);
  string reloc_mode = string(argv[6]);
  string output_name = string(argv[7]);

  if (path_to_dataset.back() != '/')
    path_to_dataset += "/";

  string output_folder = output_name;
  if (output_folder.back() != '/')
    output_folder += "/";
  fs::create_directories(output_folder);

  // Load associations
  vector<string> vstrImageFilenamesRGB;
  vector<string> vstrImageFilenamesD;
  vector<double> vTimestamps;
  string strAssociationFile = path_to_dataset + "associations.txt";
  LoadAssociations(strAssociationFile, vstrImageFilenamesRGB,
                   vstrImageFilenamesD, vTimestamps);

  size_t nImages = vstrImageFilenamesRGB.size();
  if (nImages == 0) {
    cerr << "No images found in association file: " << strAssociationFile
         << endl;
    return 1;
  }
  cout << "Found " << nImages << " RGB-D image pairs" << endl;

  // Load categories to ignore
  std::vector<int> categories_to_ignore;
  if (categories_to_ignore_file != "null" &&
      categories_to_ignore_file != "NULL") {
    std::ifstream fin(categories_to_ignore_file);
    int cat_id = -1;
    while (fin >> cat_id) {
      categories_to_ignore.push_back(cat_id);
      std::cout << "Ignore category: " << cat_id << std::endl;
    }
    fin.close();
  }

  // Setup relocalization mode
  auto rm = ORB_SLAM2::enumRelocalizationMode::RELOC_POINTS;
  if (reloc_mode == "objects") {
    rm = ORB_SLAM2::enumRelocalizationMode::RELOC_OBJECTS;
  } else if (reloc_mode == "points+objects") {
    rm = ORB_SLAM2::enumRelocalizationMode::RELOC_OBJECTS_POINTS;
  }

  // Create SLAM system in RGB-D mode
  ORB_SLAM2::System SLAM(vocabulary_file, parameters_file,
                         ORB_SLAM2::System::RGBD, true, false, 1);
  SLAM.SetRelocalizationMode(rm);

  // Setup detection manager
  std::unique_ptr<ORB_SLAM2::ImageDetectionsManager> detection_manager;
  std::string det_ext = get_file_extension(detections_file);
  bool detect_from_file = false;

  if (det_ext == "json") {
    detection_manager = std::make_unique<ORB_SLAM2::DetectionsFromFile>(
        detections_file, categories_to_ignore);
    detect_from_file = true;
  } else if (det_ext == "onnx") {
#ifdef USE_DNN
    if (detections_file.find("-seg") != std::string::npos ||
        detections_file.find("_seg") != std::string::npos) {
      std::cout << "Using SegmentationDetector for " << detections_file
                << std::endl;
      detection_manager = std::make_unique<ORB_SLAM2::SegmentationDetector>(
          detections_file, categories_to_ignore);
    } else {
      std::cout << "Using ObjectDetector for " << detections_file << std::endl;
      detection_manager = std::make_unique<ORB_SLAM2::ObjectDetector>(
          detections_file, categories_to_ignore);
    }
    detect_from_file = false;
#else
    cerr << "OpenCV DNN module not found. Cannot using ONNX model." << endl;
    return 1;
#endif
  } else {
    cerr << "Unknown detection file format: " << detections_file << endl;
    return 1;
  }

  // Main loop
  cv::Mat imRGB, imD;
  std::vector<ORB_SLAM2::Detection::Ptr> detections;
  std::vector<ORB_SLAM2::Detection::Ptr> cached_detections;

  for (size_t i = 0; i < nImages && !SLAM.ShouldQuit(); i++) {
    // Load images
    std::string rgb_filename = path_to_dataset + vstrImageFilenamesRGB[i];
    imRGB = cv::imread(rgb_filename, cv::IMREAD_UNCHANGED);
    imD = cv::imread(path_to_dataset + vstrImageFilenamesD[i],
                     cv::IMREAD_UNCHANGED);
    double timestamp = vTimestamps[i];

    if (imRGB.empty()) {
      cerr << "Failed to load image: " << rgb_filename << endl;
      continue;
    }
    if (imD.empty()) {
      cerr << "Failed to load depth: "
           << path_to_dataset + vstrImageFilenamesD[i] << endl;
      continue;
    }

    // Get detections
    if (detect_from_file) {
      detections = detection_manager->detect(rgb_filename);
    } else {
      // Logic for ONNX - detect every 5 frames to speed up
      if (i % 5 == 0) {
        detections = detection_manager->detect(imRGB);
        cached_detections = detections;
      } else {
        detections = cached_detections;
      }
    }

    // Track
    SLAM.TrackRGBD(imRGB, imD, timestamp, detections);

    if (i % 100 == 0) {
      cout << "Processed " << i << "/" << nImages << " frames" << endl;
    }
  }

  cout << "Finished processing all frames" << endl;

  // Stop threads and save results
  SLAM.Shutdown();
  SLAM.SaveTrajectoryTUM(output_folder + "CameraTrajectory.txt");
  SLAM.SaveKeyFrameTrajectoryTUM(output_folder + "KeyFrameTrajectory.txt");
  SLAM.SaveMapPointsOBJ(output_folder + "MapPoints.obj");
  SLAM.SaveMapObjectsTXT(output_folder + "MapObjects.txt");
  SLAM.SaveMapObjectsOBJ(output_folder + "MapObjects.obj");

  cout << "Results saved to: " << output_folder << endl;
  return 0;
}
