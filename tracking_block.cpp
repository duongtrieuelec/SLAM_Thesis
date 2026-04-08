    Track();

    // if (!mbOnlyTracking)    // if in localization-only mode, no neeed to
    // track objects
    //     break;

    /////////////////////////////////// Objects Tracking
    //////////////////////////////////////
    // Update mean depth
    if (mState == Tracking::OK) {
      Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
      double z_mean = 0.0;
      unsigned int z_nb = 0;
      for (size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP && !mCurrentFrame.mvbOutlier[i]) {
          cv::Mat mp = pMP->GetWorldPos();
          Eigen::Vector4d p(mp.at<float>(0), mp.at<float>(1), mp.at<float>(2),
                            1.0);
          Eigen::Vector3d p_cam = Rt * p;
          z_mean += p_cam[2];
          z_nb++;
        }
      }
      z_mean /= z_nb;
      // std::cout << "Mean depth = " << z_mean << "\n";
      current_mean_depth_ = z_mean;
    }

    std::cout << "Frame " << current_frame_idx_ << " ===========\n";
    // std::cout << "Created new KF: " << createdNewKeyFrame_ << "\n";
    std::cout << "Nb Object Tracks: " << objectTracks_.size() << "\n";
    std::cout << "Nb Map Objects  : " << mpMap->GetNumberMapObjects() << "\n";
    std::cout << "Nb Detections   : " << current_frame_detections_.size()
              << " (good: " << current_frame_good_detections_.size() << ")"
              << " | State: "
              << (mState == OK ? "OK" : (mState == LOST ? "LOST" : "OTHER"))
              << "\n"
              << std::flush;
    // for (auto tr : objectTracks_) {
    //     std::cout << "    - tr " << tr->GetId() << " : " <<
    //     tr->GetNbObservations() << "\n";
    // }

    double MIN_2D_IOU_THRESH = 0.2;
    double MIN_3D_IOU_THRESH = 0.3;
    int TIME_DIFF_THRESH = 30;

    BBox2 img_bbox(0, 0, im.cols, im.rows);

    if (mState == Tracking::OK) {

      // Keep only detections with a certain score
      if (current_frame_good_detections_.size() != 0) {

        KeyFrame *kf = mpLastKeyFrame;
        if (!createdNewKeyFrame_)
          kf = nullptr;

        // pre-compute all the projections of all ellipsoids which already
        // reconstructed
        Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
        Matrix34d P;
        P = K_ * Rt;
        std::unordered_map<ObjectTrack::Ptr, BBox2> proj_bboxes;
        for (auto tr : objectTracks_) {
          if (tr->GetStatus() == ObjectTrackStatus::INITIALIZED ||
              tr->GetStatus() == ObjectTrackStatus::IN_MAP) {
            MapObject *obj = tr->GetMapObject();
            Eigen::Vector3d c = obj->GetEllipsoid().GetCenter();
            double z = Rt.row(2).dot(c.homogeneous());
            auto ell = obj->GetEllipsoid().project(P);
            BBox2 bb = ell.ComputeBbox();
            if (bboxes_intersection(bb, img_bbox) < 0.3 * bbox_area(bb)) {
              continue;
            }
            proj_bboxes[tr] = ell.ComputeBbox();

            // Check occlusions and keep only the nearest
            std::unordered_set<ObjectTrack::Ptr> hidden;
            for (auto it : proj_bboxes) {
              if (it.first != tr && bboxes_iou(it.second, bb) > 0.9) {
                Eigen::Vector3d c2 =
                    it.first->GetMapObject()->GetEllipsoid().GetCenter();
                double z2 = Rt.row(2).dot(c2.homogeneous());
                if (z < z2) {
                  // remove z2
                  hidden.insert(it.first);
                } else {
                  // remove z
                  hidden.insert(tr);
                }
                break;
              }
            }
            for (auto hid : hidden) {
              proj_bboxes.erase(hid);
            }
          }
        }

        // find possible tracks
        std::vector<ObjectTrack::Ptr> possible_tracks;
        for (auto tr : objectTracks_) {
          auto bb = tr->GetLastBbox();
          if (tr->GetLastObsFrameId() + 60 >= current_frame_idx_ &&
              bboxes_intersection(bb, img_bbox) >= 0.3 * bbox_area(bb)) {
            possible_tracks.push_back(tr);
          } else if (proj_bboxes.find(tr) != proj_bboxes.end()) {
            possible_tracks.push_back(tr);
          }
        }

        // Associated map points to each detection
        std::vector<std::unordered_set<MapPoint *>> assoc_map_points(
            current_frame_good_detections_.size());
        for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
          for (size_t j = 0; j < mCurrentFrame.mvKeysUn.size(); ++j) {
            if (mCurrentFrame.mvpMapPoints[j]) {
              const auto &kp = mCurrentFrame.mvKeysUn[j];
              MapPoint *corresp_map_point = mCurrentFrame.mvpMapPoints[j];
              if (is_inside_bbox(kp.pt.x, kp.pt.y,
                                 current_frame_good_detections_[i]->bbox)) {
                assoc_map_points[i].insert(corresp_map_point);
              }
            }
          }
        }

        // Try to match detections to existing object track based on the
        // associated map points
        int THRESHOLD_NB_MATCH = 10;
        std::vector<int> matched_by_points(
            current_frame_good_detections_.size(), -1);
        std::vector<std::vector<size_t>> nb_matched_points(
            current_frame_good_detections_.size(), std::vector<size_t>());
        for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
          int det_cat = current_frame_good_detections_[i]->category_id;
          size_t max_nb_matches = 0;
          size_t best_matched_track = 0;
          for (size_t j = 0; j < possible_tracks.size(); ++j) {
            auto tr_map_points = possible_tracks[j]->GetAssociatedMapPoints();
            size_t n =
                count_set_map_intersection(assoc_map_points[i], tr_map_points);
            if (n > max_nb_matches) {
              max_nb_matches = n;
              best_matched_track = j;
            }

            if (det_cat != possible_tracks[j]->GetCategoryId())
              n = 0;
            nb_matched_points[i].push_back(n);
          }

          if (max_nb_matches > THRESHOLD_NB_MATCH &&
              current_frame_good_detections_[i]->category_id ==
                  possible_tracks[best_matched_track]->GetCategoryId()) {
            matched_by_points[i] = best_matched_track;
          }
        }

        int m = std::max(possible_tracks.size(),
                         current_frame_good_detections_.size());
        dlib::matrix<long> cost = dlib::zeros_matrix<long>(m, m);
        std::vector<long> assignment(
            m,
            std::numeric_limits<long>::max()); // Important to have it in
                                               // 'long', max_int is used to
                                               // force assignment of tracks
                                               // already matched using points
        if (current_frame_good_detections_.size() > 0) {
          // std::cout << "Hungarian algorithm size " << m << "\n";
          for (size_t di = 0; di < current_frame_good_detections_.size();
               ++di) {
            auto det = current_frame_good_detections_[di];

            for (size_t ti = 0; ti < possible_tracks.size(); ++ti) {
              auto tr = possible_tracks[ti];
              if (tr->GetCategoryId() == det->category_id) {
                double iou_2d = 0;
                double iou_3d = 0;

                if (tr->GetLastObsFrameId() + TIME_DIFF_THRESH >=
                    current_frame_idx_)
                  iou_2d = bboxes_iou(tr->GetLastBbox(), det->bbox);

                if (proj_bboxes.find(tr) != proj_bboxes.end())
                  iou_3d = bboxes_iou(proj_bboxes[tr], det->bbox);

                if (iou_2d < MIN_2D_IOU_THRESH)
                  iou_2d = 0;
                if (iou_3d < MIN_3D_IOU_THRESH)
                  iou_3d = 0;

                // std::cout << "2D: " << iou_2d << "\n";
                // std::cout << "3D: " << iou_3d << "\n";
                cost(di, ti) = std::max(iou_2d, iou_3d) * 1000;
              }
            }

            if (matched_by_points[di] != -1) {
              cost(di, matched_by_points[di]) = std::numeric_limits<int>::max();
            }
          }

          // for (size_t i = 0; i < current_frame_good_detections_.size(); ++i)
          // {
          //     for (size_t j = 0; j < possible_tracks.size(); ++j) {
          //         // std::cout << i << " " << j << " " <<
          //         nb_matched_points[i][j] << "\n"; cost(i, j) +=
          //         nb_matched_points[i][j] * 1000;
          //     }
          // }

          assignment = dlib::max_cost_assignment(cost); // solve
        }

        std::vector<ObjectTrack::Ptr> new_tracks;
        for (size_t di = 0; di < current_frame_good_detections_.size(); ++di) {
          auto det = current_frame_good_detections_[di];
          auto assigned_track_idx = assignment[di];
          if (assigned_track_idx >= static_cast<long>(possible_tracks.size()) ||
              cost(di, assigned_track_idx) == 0) {
            // assigned to non-existing => means not assigned
            auto tr = ObjectTrack::CreateNewObjectTrack(
                det->category_id, det->bbox, det->score, Rt, current_frame_idx_,
                this, kf);
            // std::cout << "create new track " << tr->GetId() << "\n";
            new_tracks.push_back(tr);
          } else {
            ObjectTrack::Ptr associated_track =
                possible_tracks[assigned_track_idx];
            associated_track->AddDetection(det->bbox, det->score, Rt,
                                           current_frame_idx_, kf);
            if (kf &&
                associated_track->GetStatus() == ObjectTrackStatus::IN_MAP) {
              // std::cout << "Add modified objects" << std::endl;
              if (local_object_mapper_)
                local_object_mapper_->InsertModifiedObject(
                    associated_track->GetMapObject());
            }
          }
        }

        for (auto tr : new_tracks)
          objectTracks_.push_back(tr);

        if (!mbOnlyTracking) {
          for (auto &tr : objectTracks_) {
            if (tr->GetLastObsFrameId() == current_frame_idx_) {
              // Try reconstruct from points
              if ((tr->GetNbObservations() > 4 &&
                   tr->GetStatus() == ObjectTrackStatus::ONLY_2D) ||
                  (tr->GetNbObservations() % 2 == 0 &&
                   tr->GetStatus() == ObjectTrackStatus::INITIALIZED)) {
                // tr->ReconstructFromSamplesEllipsoid();
                // tr->ReconstructFromSamplesCenter();

                // Debug log added
                std::cout << "Attempting ReconstructFromCenter for track id: "
                          << tr->GetId() << " Obs: " << tr->GetNbObservations()
                          << std::endl;
                bool status_rec =
                    tr->ReconstructFromCenter(); // try to reconstruct and
                                                 // change status to INITIALIZED
                                                 // if success
                // tr->ReconstructFromLandmarks(mpMap);
                // tr->ReconstructCrocco(false); // not working
                if (status_rec)
                  tr->OptimizeReconstruction(mpMap);
              }
            }

            // Try to optimize objects and insert in the map
            // Lower threshold from 40 to 15 to help ellipsoids appear sooner
            if (tr->GetNbObservations() >= 15 &&
                tr->GetStatus() == ObjectTrackStatus::INITIALIZED) {
              tr->OptimizeReconstruction(mpMap);
              // std::cout << "First opimitzation done.\n";
              auto checked = tr->CheckReprojectionIoU(0.3);
              // std::cout << "Check reprojection " << checked << ".\n";
              if (checked) {
                // Add object to map
                tr->InsertInMap(mpMap);
                // Add object in the local object mapping thread to run a fusion
                // checking
                if (local_object_mapper_)
                  local_object_mapper_->InsertModifiedObject(
                      tr->GetMapObject());
              } else {
                tr->SetIsBad(); // or only reset to ONLY_2D ?
              }
            }
          }
        }
      }

      if (!mbOnlyTracking) {
        // Remove objects that are not tracked anymore and not initialized or in
        // the map
        for (ObjectTrack::Ptr tr : objectTracks_) {
          if (static_cast<int>(tr->GetLastObsFrameId()) <
                  static_cast<int>(current_frame_idx_) - TIME_DIFF_THRESH &&
              tr->GetStatus() != ObjectTrackStatus::IN_MAP) {
            tr->SetIsBad();
          }
        }

        // Clean bad objects
        auto tr_it = objectTracks_.begin();
        while (tr_it != objectTracks_.end()) {
          auto temp = *tr_it;
          ++tr_it;
          if (temp->IsBad())
            RemoveTrack(temp);
        }
      }
    }

    // std::cout << "Object Tracks: " << objectTracks_.size() << "\n";
    mpFrameDrawer->Update(this);
  }

  if (mpARViewer) { // Update AR viewer camera
    mpARViewer->UpdateFrame(im_rgb_);
    if (mCurrentFrame.mTcw.rows == 4)
      mpARViewer->SetCurrentCameraPose(
          cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw));
  }

  return mCurrentFrame.mTcw.clone();
}

// GrabImageMonocular with segmentation mask for person keypoint filtering
cv::Mat
Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp,
                             const std::vector<Detection::Ptr> &detections,
                             bool force_relocalize,
                             const cv::Mat &person_mask) {
  // Convert to grayscale
  if (im.channels() == 3) {
    if (mbRGB)
      cvtColor(im, mImGray, cv::COLOR_RGB2GRAY);
    else
      cvtColor(im, mImGray, cv::COLOR_BGR2GRAY);
  } else if (im.channels() == 4) {
    if (mbRGB)
      cvtColor(im, mImGray, cv::COLOR_RGBA2GRAY);
    else
      cvtColor(im, mImGray, cv::COLOR_BGRA2GRAY);
  } else {
    mImGray = im;
  }

  if (im.channels() == 3)
    im_rgb_ = im.clone();
  else
    cvtColor(im, im_rgb_, cv::COLOR_GRAY2RGB);

  // Create Frame
  if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) {
    mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor,
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
  } else {
    mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft,
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
  }

  current_frame_detections_ = detections;
  current_frame_good_detections_.clear();

  std::vector<Eigen::Vector4d> person_bboxes;
  for (auto det : current_frame_detections_) {
    // Lower threshold to capture objects with lower confidence scores
    if (det->score > 0.2) {
      current_frame_good_detections_.push_back(det);
    }
  }

  // Apply segmentation mask filtering (more precise than bounding box)
  // Filter even during initialization to avoid tracking dynamic objects
  if (!person_mask.empty()) {
    mCurrentFrame.FilterKeypointsWithMask(person_mask);
  }

  // Feature Weighting for prioritized objects (e.g., TV/Monitor - class 62)
  if (!current_frame_good_detections_.empty()) {
    cv::Mat weight_mask = cv::Mat::zeros(im.rows, im.cols, CV_32F);

    bool found_target = false;
    for (auto det : current_frame_good_detections_) {
      if (det->category_id == 62) { // TV / Monitor in COCO
        int x1 = (int)det->bbox[0];
        int y1 = (int)det->bbox[1];
        int x2 = (int)det->bbox[2];
        int y2 = (int)det->bbox[3];

        // Clip to image bounds
        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(im.cols, x2);
        y2 = std::min(im.rows, y2);

        if (x2 > x1 && y2 > y1) {
          weight_mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)) = 100.0f;
          found_target = true;
        }
      }
    }

    if (found_target) {
      mCurrentFrame.AssignFeatureWeights(weight_mask);
    }
  }

  if (force_relocalize) {
    // ... relocalization logic would go here but we just call Track() instead
    Track();
    mpFrameDrawer->Update(this);
    if (mCurrentFrame.mTcw.rows && mCurrentFrame.mTcw.cols)
      mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
  } else {
    Track();
    current_frame_idx_++;
    createdNewKeyFrame_ = false;
    mpFrameDrawer->Update(this);
  }

  double MIN_2D_IOU_THRESH = 0.2;
  double MIN_3D_IOU_THRESH = 0.3;
  int TIME_DIFF_THRESH = 30;

  BBox2 img_bbox(0, 0, im.cols, im.rows);

  if (mState == Tracking::OK) {

    // Keep only detections with a certain score
    if (current_frame_good_detections_.size() != 0) {

      KeyFrame *kf = mpLastKeyFrame;
      if (!createdNewKeyFrame_)
        kf = nullptr;

      // pre-compute all the projections of all ellipsoids which already
      // reconstructed
      Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
      Matrix34d P;
      P = K_ * Rt;
      std::unordered_map<ObjectTrack::Ptr, BBox2> proj_bboxes;
      for (auto tr : objectTracks_) {
        if (tr->GetStatus() == ObjectTrackStatus::INITIALIZED ||
            tr->GetStatus() == ObjectTrackStatus::IN_MAP) {
          MapObject *obj = tr->GetMapObject();
          Eigen::Vector3d c = obj->GetEllipsoid().GetCenter();
          double z = Rt.row(2).dot(c.homogeneous());
          auto ell = obj->GetEllipsoid().project(P);
          BBox2 bb = ell.ComputeBbox();
          if (bboxes_intersection(bb, img_bbox) < 0.3 * bbox_area(bb)) {
            continue;
          }
          proj_bboxes[tr] = ell.ComputeBbox();

          // Check occlusions and keep only the nearest
          std::unordered_set<ObjectTrack::Ptr> hidden;
          for (auto it : proj_bboxes) {
            if (it.first != tr && bboxes_iou(it.second, bb) > 0.9) {
              Eigen::Vector3d c2 =
                  it.first->GetMapObject()->GetEllipsoid().GetCenter();
              double z2 = Rt.row(2).dot(c2.homogeneous());
              if (z < z2) {
                // remove z2
                hidden.insert(it.first);
              } else {
                // remove z
                hidden.insert(tr);
              }
              break;
            }
          }
          for (auto hid : hidden) {
            proj_bboxes.erase(hid);
          }
        }
      }

      // find possible tracks
      std::vector<ObjectTrack::Ptr> possible_tracks;
      for (auto tr : objectTracks_) {
        auto bb = tr->GetLastBbox();
        if (tr->GetLastObsFrameId() + 60 >= current_frame_idx_ &&
            bboxes_intersection(bb, img_bbox) >= 0.3 * bbox_area(bb)) {
          possible_tracks.push_back(tr);
        } else if (proj_bboxes.find(tr) != proj_bboxes.end()) {
          possible_tracks.push_back(tr);
        }
      }

      // Associated map points to each detection
      std::vector<std::unordered_set<MapPoint *>> assoc_map_points(
          current_frame_good_detections_.size());
      for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
        for (size_t j = 0; j < mCurrentFrame.mvKeysUn.size(); ++j) {
          if (mCurrentFrame.mvpMapPoints[j]) {
            const auto &kp = mCurrentFrame.mvKeysUn[j];
            MapPoint *corresp_map_point = mCurrentFrame.mvpMapPoints[j];
            if (is_inside_bbox(kp.pt.x, kp.pt.y,
                               current_frame_good_detections_[i]->bbox)) {
              assoc_map_points[i].insert(corresp_map_point);
            }
          }
        }
      }

      // Try to match detections to existing object track based on the
      // associated map points
      int THRESHOLD_NB_MATCH = 10;
      std::vector<int> matched_by_points(current_frame_good_detections_.size(),
                                         -1);
      std::vector<std::vector<size_t>> nb_matched_points(
          current_frame_good_detections_.size(), std::vector<size_t>());
      for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
        int det_cat = current_frame_good_detections_[i]->category_id;
        size_t max_nb_matches = 0;
        size_t best_matched_track = 0;
        for (size_t j = 0; j < possible_tracks.size(); ++j) {
          auto tr_map_points = possible_tracks[j]->GetAssociatedMapPoints();
          size_t n =
              count_set_map_intersection(assoc_map_points[i], tr_map_points);
          if (n > max_nb_matches) {
            max_nb_matches = n;
            best_matched_track = j;
          }

          if (det_cat != possible_tracks[j]->GetCategoryId())
            n = 0;
          nb_matched_points[i].push_back(n);
        }

        if (max_nb_matches > THRESHOLD_NB_MATCH &&
            current_frame_good_detections_[i]->category_id ==
                possible_tracks[best_matched_track]->GetCategoryId()) {
          matched_by_points[i] = best_matched_track;
        }
      }

      int m = std::max(possible_tracks.size(),
                       current_frame_good_detections_.size());
      dlib::matrix<long> cost = dlib::zeros_matrix<long>(m, m);
      std::vector<long> assignment(
          m,
          std::numeric_limits<long>::max()); // Important to have it in
                                             // 'long', max_int is used to
                                             // force assignment of tracks
                                             // already matched using points
      if (current_frame_good_detections_.size() > 0) {
        // std::cout << "Hungarian algorithm size " << m << "\n";
        for (size_t di = 0; di < current_frame_good_detections_.size(); ++di) {
          auto det = current_frame_good_detections_[di];

          for (size_t ti = 0; ti < possible_tracks.size(); ++ti) {
            auto tr = possible_tracks[ti];
            if (tr->GetCategoryId() == det->category_id) {
              double iou_2d = 0;
              double iou_3d = 0;

              if (tr->GetLastObsFrameId() + TIME_DIFF_THRESH >=
                  current_frame_idx_)
                iou_2d = bboxes_iou(tr->GetLastBbox(), det->bbox);

              if (proj_bboxes.find(tr) != proj_bboxes.end())
                iou_3d = bboxes_iou(proj_bboxes[tr], det->bbox);

              if (iou_2d < MIN_2D_IOU_THRESH)
                iou_2d = 0;
              if (iou_3d < MIN_3D_IOU_THRESH)
                iou_3d = 0;

              // std::cout << "2D: " << iou_2d << "\n";
              // std::cout << "3D: " << iou_3d << "\n";
              cost(di, ti) = std::max(iou_2d, iou_3d) * 1000;
            }
          }

          if (matched_by_points[di] != -1) {
            cost(di, matched_by_points[di]) = std::numeric_limits<int>::max();
          }
        }

        // for (size_t i = 0; i < current_frame_good_detections_.size(); ++i)
        // {
        //     for (size_t j = 0; j < possible_tracks.size(); ++j) {
        //         // std::cout << i << " " << j << " " <<
        //         nb_matched_points[i][j] << "\n"; cost(i, j) +=
        //         nb_matched_points[i][j] * 1000;
        //     }
        // }

        assignment = dlib::max_cost_assignment(cost); // solve
      }

      std::vector<ObjectTrack::Ptr> new_tracks;
      for (size_t di = 0; di < current_frame_good_detections_.size(); ++di) {
        auto det = current_frame_good_detections_[di];
        auto assigned_track_idx = assignment[di];
        if (assigned_track_idx >= static_cast<long>(possible_tracks.size()) ||
            cost(di, assigned_track_idx) == 0) {
          // assigned to non-existing => means not assigned
          auto tr = ObjectTrack::CreateNewObjectTrack(
              det->category_id, det->bbox, det->score, Rt, current_frame_idx_,
              this, kf);
          // std::cout << "create new track " << tr->GetId() << "\n";
          new_tracks.push_back(tr);
        } else {
          ObjectTrack::Ptr associated_track =
              possible_tracks[assigned_track_idx];
          associated_track->AddDetection(det->bbox, det->score, Rt,
                                         current_frame_idx_, kf);
          if (kf &&
              associated_track->GetStatus() == ObjectTrackStatus::IN_MAP) {
            // std::cout << "Add modified objects" << std::endl;
            if (local_object_mapper_)
              local_object_mapper_->InsertModifiedObject(
                  associated_track->GetMapObject());
          }
        }
      }

      for (auto tr : new_tracks)
        objectTracks_.push_back(tr);

      if (!mbOnlyTracking) {
        for (auto &tr : objectTracks_) {
          if (tr->GetLastObsFrameId() == current_frame_idx_) {
            // Try reconstruct from points
            if ((tr->GetNbObservations() > 4 &&
                 tr->GetStatus() == ObjectTrackStatus::ONLY_2D) ||
                (tr->GetNbObservations() % 2 == 0 &&
                 tr->GetStatus() == ObjectTrackStatus::INITIALIZED)) {
              // tr->ReconstructFromSamplesEllipsoid();
              // tr->ReconstructFromSamplesCenter();

              bool status_rec =
                  tr->ReconstructFromCenter(); // try to reconstruct and
                                               // change status to INITIALIZED
                                               // if success
              // tr->ReconstructFromLandmarks(mpMap);
              // tr->ReconstructCrocco(false); // not working
              if (status_rec)
                tr->OptimizeReconstruction(mpMap);
            }
          }

          // Try to optimize objects and insert in the map
          // Lower threshold from 40 to 15 to help ellipsoids appear sooner
          if (tr->GetNbObservations() >= 15 &&
              tr->GetStatus() == ObjectTrackStatus::INITIALIZED) {
            tr->OptimizeReconstruction(mpMap);
            // std::cout << "First opimitzation done.\n";
            auto checked = tr->CheckReprojectionIoU(0.3);
            // std::cout << "Check reprojection " << checked << ".\n";
            if (checked) {
              // Add object to map
              tr->InsertInMap(mpMap);
              // Add object in the local object mapping thread to run a fusion
              // checking
              if (local_object_mapper_)
                local_object_mapper_->InsertModifiedObject(tr->GetMapObject());
            } else {
              tr->SetIsBad(); // or only reset to ONLY_2D ?
            }
          }
        }
      }
    }

    if (!mbOnlyTracking) {
      // Remove objects that are not tracked anymore and not initialized or in
      // the map
      for (ObjectTrack::Ptr tr : objectTracks_) {
        if (static_cast<int>(tr->GetLastObsFrameId()) <
                static_cast<int>(current_frame_idx_) - TIME_DIFF_THRESH &&
            tr->GetStatus() != ObjectTrackStatus::IN_MAP) {
          tr->SetIsBad();
        }
      }

      // Clean bad objects
      auto tr_it = objectTracks_.begin();
      while (tr_it != objectTracks_.end()) {
        auto temp = *tr_it;
        ++tr_it;
        if (temp->IsBad())
          RemoveTrack(temp);
      }
    }
  }

  // std::cout << "Object Tracks: " << objectTracks_.size() << "\n";
  std::cout << "[Tracking] Track End. State: " << mState
            << ". FrameID: " << mCurrentFrame.mnId << std::endl;
  mpFrameDrawer->Update(this);

  if (mpARViewer) {
    mpARViewer->UpdateFrame(im_rgb_);
    if (mCurrentFrame.mTcw.rows == 4)
      mpARViewer->SetCurrentCameraPose(
          cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw));
  }

  return mCurrentFrame.mTcw.clone();
