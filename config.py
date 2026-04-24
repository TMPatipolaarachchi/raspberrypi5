"""
Configuration settings for Elephant Detection and Behavior Classification System
Optimized for Raspberry Pi 5
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
ELEPHANT_DETECT_MODEL = BASE_DIR / "elephent detect" / "model" / "best.pt"
POSE_MODEL_DIR = BASE_DIR / "posed based" / "model"
SOUND_MODEL_DIR = BASE_DIR / "sound-based" / "model"

# Model files
POSE_YOLO_MODEL = POSE_MODEL_DIR / "yolov8n-pose.pt"
POSE_CLASSIFIER = POSE_MODEL_DIR / "pose_model.pkl"
POSE_SCALER = POSE_MODEL_DIR / "pose_scaler.pkl"
POSE_LABEL_ENCODER = POSE_MODEL_DIR / "pose_label_encoder.pkl"
POSE_FEATURE_ORDER = POSE_MODEL_DIR / "pose_feature_order.json"

SOUND_CNN_LSTM_MODEL = SOUND_MODEL_DIR / "cnn_lstm_model.h5"
SOUND_RF_MODEL = SOUND_MODEL_DIR / "rf_model.pkl"
SOUND_XGB_MODEL = SOUND_MODEL_DIR / "xgb_model.pkl"
SOUND_SCALER = SOUND_MODEL_DIR / "scaler.pkl"
SOUND_HYBRID_SCALER = SOUND_MODEL_DIR / "hybrid_scaler.pkl"
SOUND_LABEL_ENCODER = SOUND_MODEL_DIR / "label_encoder.pkl"
SOUND_FEATURE_ORDER = SOUND_MODEL_DIR / "feature_order.json"
SOUND_HYBRID_FEATURE_ORDER = SOUND_MODEL_DIR / "hybrid_feature_order.json"

# Detection settings
DETECTION_CONFIDENCE = 0.5
DETECTION_IOU_THRESHOLD = 0.45

# Pose estimation settings
POSE_CONFIDENCE = 0.3
POSE_IOU_THRESHOLD = 0.5

# Audio settings
AUDIO_SAMPLE_RATE = 22050
AUDIO_DURATION = 3.0  # seconds per segment
AUDIO_HOP_LENGTH = 512
AUDIO_N_FFT = 2048
AUDIO_N_MFCC = 13

# Video processing settings
FRAME_SKIP = 5  # Process every nth frame for efficiency on Pi
MAX_RESOLUTION = (640, 480)  # Max resolution for Pi processing

# Behavior classification settings
BEHAVIOR_CONFIDENCE_THRESHOLD = 0.6
BEHAVIOR_SMOOTHING_WINDOW = 5  # Number of frames for temporal smoothing

# Group classification thresholds
GROUP_INDIVIDUAL_MAX = 1
GROUP_FAMILY_MIN_ADULTS = 2
GROUP_FAMILY_MAX_ADULTS = 2
GROUP_FAMILY_MIN_CALVES = 1

# Raspberry Pi optimization
ENABLE_GPU = False  # Raspberry Pi doesn't have CUDA GPU
NUM_THREADS = 4  # Raspberry Pi 5 has 4 cores
USE_LITE_MODELS = True  # Use lighter models when available

# Output settings
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
SAVE_ANNOTATED_VIDEO = True
SAVE_JSON_RESULTS = True

# Class mappings
ELEPHANT_CLASSES = {
    0: "adult",
    1: "calf"
}

BEHAVIOR_CLASSES = {
    0: "calm",
    1: "aggressive"
}

# Behavior label normalization mapping
# Maps various label formats (case-insensitive keys) to the ONLY two allowed outputs:
#   "aggressive"  or  "calm"
# "Normal" is explicitly mapped to "calm" — no retraining needed.
# "unknown" is NEVER used as an output label anywhere in the system.
BEHAVIOR_LABEL_MAPPING = {
    # Aggressive variations
    "aggressive": "aggressive",
    "agitated":   "aggressive",
    "threat":     "aggressive",
    "threatening": "aggressive",
    "trumpet":    "aggressive",
    "trumpeting": "aggressive",
    "charging":   "aggressive",
    "attack":     "aggressive",
    # Calm / Normal variations — "Normal" from legacy models maps to "calm"
    "calm":    "calm",
    "normal":  "calm",   # KEY: old models that output "Normal" are handled here
    "relaxed": "calm",
    "passive": "calm",
    "feeding": "calm",
    "resting": "calm",
    "neutral": "calm",
}


def normalize_behavior_label(label) -> str:
    """
    Normalize ANY behavior label to one of exactly two allowed outputs:
        "aggressive"  or  "calm"

    Rules (case-insensitive, whitespace-stripped):
    - "Normal", "normal"        -> "calm"   (legacy model label)
    - "Calm", "calm", "CALM"   -> "calm"
    - "Aggressive", "aggressive" -> "aggressive"
    - Any unrecognised / None / non-string input -> "calm"  (safe default)

    "unknown" is NEVER returned by this function.

    Args:
        label: Raw behavior label string (or any value)

    Returns:
        "aggressive" or "calm"  — always, guaranteed.
    """
    if not label or not isinstance(label, str):
        return "calm"
    normalized = BEHAVIOR_LABEL_MAPPING.get(label.strip().lower())
    if normalized:
        return normalized
    # Unrecognised label: default to "calm" (safe fallback — never "unknown")
    return "calm"


GROUP_CLASSIFICATIONS = ["individual", "family", "herd"]
