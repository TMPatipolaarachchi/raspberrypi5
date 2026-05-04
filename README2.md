# Elephant Detection and Behavior Classification Project Report Guide

## 1. Project Idea
This project is an edge-based elephant monitoring system designed to detect elephants from video, classify their group type and behavior, and trigger safe response actions. The system is built mainly for Raspberry Pi 5 and uses computer vision, audio analysis, and lightweight device communication to support human-elephant conflict mitigation.

The core idea is simple:
- Detect whether elephants are present in a video or camera stream.
- Count the elephants and classify them as adult or calf.
- Classify the group as individual, family, or herd.
- Predict elephant behavior as aggressive or calm using pose and sound models.
- Send the final result to an ESP32 through serial communication.
- Activate a bee-sound deterrent only when the conditions are safe.

This document can be used as a base for writing the final report because it explains the full project structure, the purpose of each file, the workflow, the models used, and the output actions.

## 2. Project Objective
The main objective of the project is to build an intelligent wildlife monitoring and alert system that works locally on a Raspberry Pi. It aims to detect elephants early, analyze their behavior, and respond in a way that is practical, low-cost, and suitable for field deployment.

The project also demonstrates how multiple AI models can be combined into one pipeline:
- Object detection for elephant presence.
- Pose-based behavior classification.
- Sound-based behavior classification.
- Decision fusion and temporal smoothing.
- Device output through serial and audio deterrent modules.

## 3. System Summary
The system follows this runtime flow:

1. Input is taken from either a video file or a live camera feed.
2. Elephant detection identifies elephants and counts adults and calves.
3. Pose classifier predicts behavior from body keypoints.
4. Sound classifier predicts behavior from extracted audio.
5. Integrated pipeline combines the predictions into one final behavior label.
6. The final result is printed, logged, and passed to output modules.
7. ESP32 serial integration sends a status payload.
8. Bee sound deterrent activates only when the rule set allows it.

The system is optimized for edge processing so that it can run without cloud dependency.

## 4. Main Features
- Elephant detection using a custom YOLOv8 model.
- Adult and calf classification.
- Group classification into individual, family, or herd.
- Pose-based behavior detection using keypoints and an ML classifier.
- Sound-based behavior detection using CNN-LSTM and classical ML models.
- Fusion of pose and sound predictions.
- Temporal smoothing to reduce unstable predictions.
- Raspberry Pi optimization for performance.
- Serial communication with ESP32.
- Bee sound deterrent playback through a Bluetooth speaker.
- Optional annotated video and JSON result saving.

## 5. Repository Structure

### Core application files
- `main.py` - Main entry point. It handles arguments, model checks, setup, processing, and final output.
- `integrated_pipeline.py` - Orchestrates elephant detection, pose classification, sound classification, fusion, smoothing, and result aggregation.
- `config.py` - Stores model paths, thresholds, constants, behavior mappings, and output settings.

### AI and inference modules
- `elephant_detector.py` - Detects elephants and classifies them as adult or calf.
- `pose_classifier.py` - Extracts pose keypoints and predicts aggressive or calm behavior.
- `sound_classifier.py` - Extracts audio features and predicts behavior using CNN-LSTM and machine learning ensembles.
- `diagnose_sound_models.py` - Diagnostic helper for checking audio model output and consistency.

### Output and response modules
- `esp32_serial_integration.py` - Validates the final result and sends a JSON payload to ESP32 over UART serial.
- `elephant_bee_sound_raspberry.py` - Plays bee sounds through a Bluetooth speaker when activation rules are satisfied.
- `alert_system.py` - General-purpose alert manager for GPIO, webhook, and email notifications.

### Camera and optimization
- `raspberry_pi_camera_input.py` - Camera input helper for Raspberry Pi camera and OpenCV fallback.
- `pi_optimizer.py` - Performance tuning utilities for Raspberry Pi execution.

### Deployment and setup
- `setup_raspberry_pi.sh` - Setup helper for installing dependencies and preparing the Pi.
- `elephant-detection.service` - Systemd service file for running the application automatically.
- `requirements.txt` - Main Python dependencies.
- `README.md` - Main project documentation.
- `README1.md` - Extended understanding and website blueprint.

### Supporting documentation
- `CHANGES_BY_FILE.md` - Summary of file-level changes.
- `DEBUGGING_REPORT.md` - Notes on debugging and fixes.
- `FIX_SUMMARY.md` - Quick summary of implemented fixes.
- `CONTRIBUTORS.md` - Contributor information.

### Model and asset folders
- `elephent detect/model/` - Elephant detection model files, including `best.pt`.
- `posed based/model/` - Pose model files, metadata, and pose classifier support files.
- `sound-based/model/` - Sound model files, scalers, encoders, and feature-order metadata.
- `bee_sounds/` - Audio clips used for bee-sound deterrence.
- `logs/` - Runtime logs.

### Other files and artifacts
- `video.mp4` - Sample media file for testing.
- `Website.pdf` - Additional document in the repository.
- `nano.11606.save` - Editor save artifact.
- `__pycache__/` - Python cache folder.

## 6. How the Full Pipeline Works

### 6.1 Input stage
The system accepts:
- A video file path.
- A live camera feed using `--camera`.

The main entry point is `main.py`, which decides whether the system runs on video or camera mode.

### 6.2 Elephant detection stage
`elephant_detector.py` loads a custom YOLOv8 model and detects elephants in each frame. It classifies each detection as:
- adult
- calf

The detector then counts the detections and assigns a group type:
- individual - one elephant
- family - two adults and at least one calf
- herd - other multi-elephant combinations

### 6.3 Pose behavior stage
`pose_classifier.py` uses a YOLOv8-pose model to extract body keypoints. It then calculates geometric features such as:
- ear spread
- trunk length
- tail length
- leg stance
- body ratios and angles

These features are passed to a trained machine learning classifier that predicts behavior.

### 6.4 Sound behavior stage
`sound_classifier.py` extracts audio from the video and computes audio features such as:
- MFCCs
- spectral centroid
- spectral bandwidth
- spectral rolloff
- zero crossing rate
- HNR-like measures

It uses two model paths:
- CNN-LSTM deep learning model
- Random Forest and XGBoost ensemble

The final sound prediction is derived from these outputs.

### 6.5 Fusion stage
`integrated_pipeline.py` combines pose and sound predictions.

The final behavior result is always normalized to one of only two labels:
- aggressive
- calm

If both pose and sound are available, the pipeline performs weighted fusion. If only one is available, it uses the valid one. If neither is available, it safely defaults to calm.

### 6.6 Temporal smoothing
The pipeline keeps a short behavior buffer across frames. This reduces flickering predictions and improves stability. The default smoothing window is defined in `config.py`.

### 6.7 Output stage
After processing a video, the system creates a final summary object containing:
- whether elephants were detected
- dominant group classification
- dominant behavior
- maximum adult count
- maximum calf count
- behavior confidence
- behavior source
- processing time

Then the final result is passed to:
- `esp32_serial_integration.py`
- `elephant_bee_sound_raspberry.py`

## 7. Important Module Roles

### `main.py`
This is the user-facing command-line interface. It:
- Parses input arguments.
- Checks whether all required models exist.
- Applies Raspberry Pi optimizations.
- Runs the pipeline.
- Prints a final report to the console.
- Sends the final detection result to ESP32 and the bee-sound module.

### `config.py`
This file contains the project constants and paths. It also standardizes behavior labels so the system only uses:
- aggressive
- calm

This is important because different models may produce legacy labels such as Normal or Threatening, and the code maps them to the standard labels.

### `integrated_pipeline.py`
This is the central controller of the system. It ties together detection, pose classification, sound classification, output formatting, and summary generation.

### `esp32_serial_integration.py`
This module validates the final result and sends a compact JSON payload to the ESP32 over `/dev/serial0` at `115200` baud.

Behavior of the payload:
- If elephants are detected, it sends a full payload with counts, group type, behavior, and timestamp.
- If no elephant is detected, it sends a minimal payload containing the pillar ID and elephant status.

### `elephant_bee_sound_raspberry.py`
This module decides whether to play a bee sound deterrent. It only activates when:
- elephant_detected is true
- behavior is calm
- group type is herd or individual
- the current time is inside the allowed playback windows

### `pi_optimizer.py`
This module improves runtime behavior on Raspberry Pi by setting optimization values for TensorFlow, OpenCV, and NumPy, and by providing warmup and frame-rate utilities.

## 8. Data and Model Details

### Elephant detection model
- File: `elephent detect/model/best.pt`
- Purpose: detect elephants in the frame.

### Pose models and metadata
- Folder: `posed based/model/`
- Includes pose detector, classifier, scaler, encoder, feature order, feature importance, and training metadata files.

### Sound models and metadata
- Folder: `sound-based/model/`
- Includes CNN-LSTM, Random Forest, XGBoost, scalers, label encoder, and feature order files.

### Bee sound assets
- Folder: `bee_sounds/`
- Used for deterrent playback.

## 9. Configuration and Rules

### Behavior normalization
All behavior labels are normalized to only:
- aggressive
- calm

### Group classification rules
- individual: exactly one elephant.
- family: two adults and at least one calf.
- herd: all other multi-elephant combinations.

### Bee sound activation rules
Bee sound plays only when all required conditions are met. It is blocked if the behavior is aggressive or the group is family.

### Output rules
The system always keeps a final behavior output and logs the decision source:
- fused
- pose_only
- sound_only
- none

## 10. Deployment View
The project is intended for local deployment on Raspberry Pi 5. The service file allows it to start automatically as a background service.

Typical deployment stack:
- Camera or video input
- Local AI inference on Raspberry Pi
- UART serial communication to ESP32
- Bluetooth speaker for deterrent playback

## 11. Limitations and Notes
These points should be mentioned in the report because they are important for accuracy:

- The current code sends alerts to ESP32 using serial communication, not LoRa or MQTT.
- The bee-sound folder must contain actual audio files for deterrent playback to work.
- The system depends on the required model files being present in the model folders.
- The project is optimized for Raspberry Pi 5, so performance may differ on weaker hardware.
- The repository contains multiple documentation files, and some descriptions are broader than the exact runtime implementation. This README2 focuses on the actual code path.

## 12. Suggested Report Structure
If you are turning this project into a formal report, use the following chapter flow:

1. Introduction
2. Problem statement
3. Project objectives
4. System architecture
5. Dataset and model description
6. Methodology
7. Implementation details
8. Results and discussion
9. Limitations and future work
10. Conclusion

## 13. Short Conclusion
This project is a complete edge AI system for elephant detection and behavior analysis. It combines multiple AI models, robust result normalization, serial communication, and a safe acoustic deterrent into a single Raspberry Pi-based workflow. README2.md can be used as a structured source for writing the final report and explaining how every part of the repository fits together.