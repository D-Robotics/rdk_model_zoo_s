English| [简体中文](./README_cn.md)


# ByteTrack

## 1. Introduction to ByteTrack

Multi-Object Tracking (MOT) aims to estimate the bounding boxes and identities of objects in videos. Most existing methods associate only detection boxes with scores above a certain threshold, discarding low-score detections (such as those of occluded objects), which leads to missed true objects and fragmented trajectories.

To address this, the ByteTrack paper proposes a simple, effective, and general association method—**BYTE (Tracking By associating Almost Every Detection Box)**—which tracks by associating almost every detection box, not just the high-score ones. For low-score detections, ByteTrack leverages their similarity to existing tracklets to recover true objects and filter out background detections.
![alt text](source/imgs/image1.png)

**Core ideas of ByteTrack:**

* **Retain almost all detection boxes:** Unlike previous methods that simply discard low-score detections, ByteTrack considers that low-score boxes may still indicate real objects (e.g., heavily occluded or motion-blurred targets).
* **Two-stage matching strategy:**
    1.  **First association:** Match high-confidence detection boxes with existing tracks, mainly relying on motion models (e.g., Kalman filter predictions) and appearance similarity (if Re-ID features are used). Unmatched tracks proceed to the next stage.
    2.  **Second association:** Match the remaining unmatched tracks (often those with low detection scores due to occlusion) with low-confidence detection boxes, mainly using IoU as the similarity metric, since appearance features of low-score boxes are often unreliable. This helps recover occluded objects and maintain trajectory continuity while filtering out background detections.
* **Track initialization:** New tracks are initialized only from unmatched high-score detection boxes.

With this refined handling of detection boxes of different scores, ByteTrack achieves state-of-the-art performance on standard MOT benchmarks (such as MOT17, MOT20), e.g., 80.3 MOTA and 77.3 IDF1 on MOT17 test set, and runs at 30 FPS on a single V100 GPU.

Original paper: [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)

![alt text](source/imgs/MOT17-01-SDP.gif)![alt text](source/imgs/MOT17-07-SDP.gif)

## 2. Quick Start

This section guides you to quickly run a pre-configured YOLO + ByteTrack demo for pedestrian detection and tracking on your RDK S100 platform. It assumes you already have the basic environment and capability to run YOLO on RDK S100.

### 2.1 Preparation

* **Hardware:** RDK S100 development board.
* **Software:**
    * Embedded Linux system for RDK S100.
    * Python 3 environment.
    * Installed `hobot_dnn` Python interface.
    * Installed basic Python libraries: `numpy`, `opencv-python`, `scipy`, etc.
    * Installed `lap` and `cython_bbox` libraries.
    * YOLO detection model (`.hbm` format, downloadable from the `ultralytics_YOLO_Detect` repo; models provided in `source/reference_hbm_models` are obtained using `ultralytics_YOLO_Detect`).
    * ByteTrack tracking code (ensure `byte_tracker.py`, `kalman_filter.py`, `matching.py`, `basetrack.py` are in the correct `tracker` path).
* **Example code:** Main script `ultralytics_YOLO_ByteTrack.py`, which integrates YOLO inference and ByteTrack tracking logic.

### 2.2 Configuration and Running

1.  **Project structure:**
    Ensure your project directory is organized as follows:
    ```
    ByteTrack/
    ├── python/
    │   └──ultralytics_YOLO_ByteTrack.py  # Main script
    ├── source/
    │   ├──hbm_models/
    │   │   └── yolo_person_model.hbm   # YOLO detection model
    │   └──track_test.mp4               # Input video for testing
    └── tracker/                        # ByteTrack core code
        ├── byte_tracker.py
        ├── kalman_filter.py
        ├── matching.py
        └── basetrack.py
    ```

    The `tracker` directory contains the core ByteTrack code. Original repo: [ByteTrack](https://github.com/FoundationVision/ByteTrack).

2.  **Run:**
    ```bash
    python3 ultralytics_YOLO_ByteTrack.py \
        --model-path ./models/yolo_person_model.hbm \
        --input ./test_video.mp4 \
        --output ./output_tracked_video.avi 
        # --score-thres 0.25  # YOLO detection confidence threshold
        # --track-thresh 0.3 
    ```
    Check the output video `output_tracked_video.avi` for tracking results.

    Tracker parameter descriptions:
    * `--score-thres`: YOLO detection confidence threshold (default: 0.25).
    * `--track-thresh`: ByteTrack track matching threshold (default: 0.3).
    * `--match-thresh`: IoU matching strictness (default: 0.7).
    * `--track-buffer`: Track lost buffer duration (default: 30).

3. **Result:** On success, an `output_tracked_video.avi` file will be generated, showing YOLO detection boxes and ByteTrack trajectories.
   
4. **Performance:** On RDK S100, ByteTrack tracker update time ≈ 2.37 ms per frame. For detection performance, refer to the `ultralytics_YOLO_Detect` repo.
   

### 2.3 Expected Results & Troubleshooting

* **Expected results:** Pedestrians in the video will be stably detected and tracked, each with a unique ID.
* **Common troubleshooting:**
    * **Few detection boxes:**
        1.  Check if YOLO's `score_thres` is too high.
        2.  **Key check:** ByteTrack's `args.track_thresh` and its internal `self.det_thresh` (i.e., `args.track_thresh + 0.1`) may be too high, causing many valid YOLO detections to be filtered out. Try lowering `args.track_thresh` (e.g., to 0.25–0.4).
    * **Frequent ID switches:** Related to `args.match_thresh` (IoU strictness) or `args.track_buffer` (lost buffer duration).

## 3. Advanced Development

Building on the quick start, you can further develop and optimize as follows:

### 3.1 Parameter Tuning

* **Systematic tuning:** Adjust YOLO detection thresholds and ByteTrack parameters (`track_thresh`, `det_thresh` (indirectly via `track_thresh`), `match_thresh`, `track_buffer`) for your specific scenario (lighting, crowd density, camera angle, object speed, etc.) to achieve the best balance (e.g., in MOTA, IDF1, or subjective visual effect).
* **Visualization:** During debugging, visualize intermediate results, such as:
    * All YOLO-detected pedestrian boxes (after applying `score_thres`).
    * High-score and low-score detection boxes filtered by ByteTrack.
    * Final trajectories output by ByteTrack.
    This helps understand the impact of parameter changes at each stage.

### 3.2 Performance Optimization (RDK S100)

* **Pre/post-processing optimization:** Check if OpenCV operations (e.g., resize, cvtColor) can be optimized, such as using more efficient interpolation or leveraging the TROS image processing library.
* **NumPy optimization:** Avoid unnecessary loops; use NumPy vectorized operations.
* **Parallel processing:** Consider running detection and tracking in separate threads (be mindful of data synchronization and Python GIL limitations).

### 3.3 Feature Extensions

* **Multi-class tracking:**
    * Currently, only pedestrians are tracked. To extend to multiple classes:
        1.  Ensure YOLO model supports multi-class detection.
        2.  Decide how to pass multi-class detections to ByteTrack:
            * **Option A (simple):** Maintain a separate `BYTETracker` instance for each class of interest. After YOLO output, dispatch detections by class ID to the corresponding tracker.
            * **Option B (integrated, may require ByteTrack modification):** Modify the `STrack` class to include a `class_id` attribute, and handle class info in `BYTETracker`'s `update` and `init_track` logic. Then, `tracker.update` output will include class IDs for each trajectory. (See Ultralytics' ByteTrack code for reference.)
* **Re-ID feature fusion (advanced):**
    * While ByteTrack's core BYTE association does not require Re-ID, combining Re-ID features can further improve performance in scenarios with long-term occlusion, low frame rates, or large movements.
    * On RDK S100, implementing Re-ID requires a lightweight Re-ID model (`.hbm` format), and modifying ByteTrack's similarity calculation and matching logic (e.g., adding Re-ID distance in the first association). This increases computational complexity and should be balanced with performance.

### 3.4 Interaction & Applications

* **Event detection:** Based on tracked trajectories, develop advanced applications such as crowd counting, area intrusion detection, or specific behavior recognition.
* **Data statistics & analysis:** Analyze tracking results for metrics like average dwell time, flow, etc.



