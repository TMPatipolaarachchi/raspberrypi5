# Elephant Detection and Behavior Classification System

A complete AI-powered system deployed on **Raspberry Pi 5** that detects elephants from video, classifies their group and behavior, sends wireless alerts via **LoRa** (with **MQTT** fallback), and activates a **bee sound deterrent** through a Bluetooth speaker to safely repel elephants.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Full Pipeline Flow](#full-pipeline-flow)
3. [Module Descriptions](#module-descriptions)
4. [Complete System Architecture](#complete-system-architecture)
5. [Directory Structure](#directory-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [LoRa and MQTT Alerts](#lora-and-mqtt-alerts)
9. [Bee Sound Deterrent](#bee-sound-deterrent)
10. [Output Format](#output-format)
11. [Running as a Service](#running-as-a-service)
12. [Alert Configuration](#alert-configuration)
13. [Performance Optimization](#performance-optimization)
14. [Model Information](#model-information)
15. [Troubleshooting](#troubleshooting)
16. [Contributors](#contributors)

---

## System Overview

This system is designed for **wildlife conservation and human-elephant conflict prevention**. A camera and microphone connected to a Raspberry Pi 5 continuously monitor an area. When elephants are detected, the system:

1. **Detects** elephants using a YOLOv8 vision model
2. **Classifies** behavior (aggressive / calm) using pose and sound AI models
3. **Classifies** the group (individual / family / herd)
4. **Sends wireless alerts** via LoRa to a remote receiver, with MQTT as automatic fallback
5. **Activates a bee sound deterrent** through a Bluetooth speaker when safe to repel elephants

Even when no elephant is detected, the system still sends a "clear" status through LoRa/MQTT so the remote receiver knows the pillar is alive and the area is safe.

---

## Full Pipeline Flow

```
VIDEO INPUT (file or camera)
        │
        ▼
┌──────────────────────────────┐
│   1. ELEPHANT DETECTION      │  elephant_detector.py
│   YOLOv8 custom model        │  → elephant_detected: True/False
│   Detects adults & calves    │  → adult_count, calf_count
│   Classifies group type      │  → group_type: individual/family/herd
└──────────────┬───────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌───────────────┐   ┌───────────────┐
│ 2a. POSE      │   │ 2b. SOUND     │
│ CLASSIFIER    │   │ CLASSIFIER    │
│ pose_         │   │ sound_        │
│ classifier.py │   │ classifier.py │
│               │   │               │
│ YOLOv8-pose   │   │ CNN-LSTM +    │
│ + ML model    │   │ RF/XGB model  │
│               │   │               │
│ Body posture  │   │ Vocalizations │
│ Keypoints     │   │ Audio features│
│ → aggressive  │   │ → aggressive  │
│   or calm     │   │   or calm     │
└──────┬────────┘   └──────┬────────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
┌──────────────────────────────┐
│  3. BEHAVIOR FUSION          │  integrated_pipeline.py
│  Weighted ensemble vote      │
│  + 5-frame temporal smooth   │
│  → Final: aggressive / calm  │
└──────────────┬───────────────┘
               │
    ┌──────────┴──────────────────────┐
    │                                 │
    ▼                                 ▼
┌────────────────────────┐   ┌────────────────────────┐
│ 4. LORA + MQTT ALERT   │   │ 5. BEE SOUND DETERRENT │
│ lora_integration.py    │   │ elephant_bee_sound_     │
│                        │   │ raspberry.py            │
│ elephant=True:         │   │                        │
│  Full payload sent     │   │ Activates ONLY when:   │
│                        │   │  • elephant=True       │
│ elephant=False:        │   │  • behavior=calm       │
│  Minimal status sent   │   │  • group=herd/individual│
│  (pillar alive + clear)│   │  • within time window  │
│                        │   │                        │
│ Transport: LoRa first  │   │ Plays bee MP3 clips    │
│ Fallback: MQTT         │   │ via Bluetooth speaker  │
│                        │   │ for 10 minutes         │
└────────────────────────┘   └────────────────────────┘
```

---

## Module Descriptions

### `main.py` — Entry Point
The command-line interface for the system. Handles argument parsing, model validation, Raspberry Pi optimization setup, and calls `IntegratedPipeline` to process a video or camera feed. After processing, it calls `process_and_send_alert()` and `process_bee_sound()` with the final result.

---

### `config.py` — Configuration
Central configuration file. Defines all file paths, model settings, detection thresholds, frame processing parameters, and the `normalize_behavior_label()` function that maps any behavior variant (e.g. "Normal", "Calm", "AGGRESSIVE") to exactly `"aggressive"` or `"calm"`.

Key settings:
```python
DETECTION_CONFIDENCE = 0.5
FRAME_SKIP = 5              # Process every 5th frame
MAX_RESOLUTION = (640, 480)
BEHAVIOR_SMOOTHING_WINDOW = 5
```

---

### `elephant_detector.py` — Elephant Detection
Uses a custom-trained **YOLOv8** model to detect elephants in a video frame. Classifies each detected elephant as adult (class 0) or calf (class 1), counts them, and assigns a group type.

**Group classification rules:**
| Condition | Group Type |
|-----------|-----------|
| Total count == 1 | `individual` |
| Adults == 2 AND calves >= 1 | `family` |
| Any other valid group | `herd` |

**Returns** a `DetectionResult` dataclass:
```python
DetectionResult(
    elephant_detected = True,
    total_count = 3,
    adult_count = 2,
    calf_count = 1,
    group_classification = "family",
    detections = [ElephantDetection(...), ...]
)
```

---

### `pose_classifier.py` — Pose-Based Behavior
Uses **YOLOv8-pose** to extract 17 body keypoints from each detected elephant. Computes geometric features (ear spread, trunk length, leg stance, body angles, etc.) and feeds them into a trained ML classifier to predict behavior.

**Output:** `PoseResult(behavior="aggressive"|"calm", confidence=0.0–1.0)`

---

### `sound_classifier.py` — Sound-Based Behavior
Extracts audio from the video using `ffmpeg`, then runs two models in an ensemble:
- **CNN-LSTM** on mel-spectrograms (deep learning)
- **Random Forest + XGBoost** on 62 hand-crafted audio features (MFCCs, spectral centroid, chroma, HNR, etc.)

**Output:** `SoundResult(behavior="aggressive"|"calm", confidence=0.0–1.0)`

---

### `integrated_pipeline.py` — Main Orchestrator
The core pipeline class `IntegratedPipeline` ties everything together:
1. Reads video frame by frame (skipping every `FRAME_SKIP` frames)
2. Runs `ElephantDetector` on each frame
3. Runs `PoseBehaviorClassifier` on each frame synchronously
4. Runs `SoundBehaviorClassifier` on the full audio asynchronously (once)
5. Fuses pose + sound predictions using weighted voting
6. Applies 5-frame temporal smoothing to stabilize behavior output
7. Aggregates all per-frame results into a final `VideoResult`

**Final `VideoResult` contains:**
```python
VideoResult(
    elephant_detected = True,
    dominant_behavior = "aggressive",         # "aggressive" or "calm" only
    dominant_group_classification = "herd",
    max_adult_count = 3,
    max_calf_count = 1,
    dominant_behavior_confidence = 0.87,
    dominant_behavior_source = "fused",       # or "pose_only", "sound_only"
    ...
)
```

---

### `lora_integration.py` — Wireless Alert System
Builds, validates, and transmits alert payloads after every inference cycle.

**Always sends — regardless of whether elephants were detected:**

| Scenario | LoRa Payload | MQTT Payload |
|----------|-------------|--------------|
| Elephant detected | Full payload: pillar_id, lat, lon, elephant=true, count, adults, calves, group, behavior, confidence, timestamp | Same as LoRa in JSON format |
| No elephant | Minimal status: pillar_id, lat, lon, elephant=false, timestamp | Same minimal payload |

**Transport logic:**
```
Try LoRa → wait for ACK (2s timeout)
    If ACK received → done, no MQTT
    If no ACK → retry (up to 3 times)
        If all retries fail → send via MQTT
            If MQTT succeeds → done
            If MQTT fails → log error, return failure status
```

**Payload formats:**

LoRa (compact key-value string):
```
PID=PILLAR_01;LAT=7.123456;LON=80.123456;EL=1;CNT=3;AD=2;CF=1;GRP=family;BEH=aggressive;CONF=0.91;TS=2026-03-08T10:30:00
```

LoRa when no elephant:
```
PID=PILLAR_01;LAT=7.123456;LON=80.123456;EL=0;TS=2026-03-08T10:30:00
```

MQTT (JSON):
```json
{
  "pillar_id": "PILLAR_01",
  "lat": 7.123456,
  "lon": 80.123456,
  "elephant": true,
  "count": 3,
  "adult": 2,
  "calf": 1,
  "group": "family",
  "behavior": "aggressive",
  "confidence": 0.91,
  "ts": "2026-03-08T10:30:00"
}
```

**Key functions:**
| Function | Purpose |
|----------|---------|
| `normalize_behavior_label(label)` | Maps any label variant to `"aggressive"` or `"calm"` |
| `normalize_group_type(...)` | Corrects group type based on actual counts |
| `validate_detection_result(result)` | Safely fills missing/invalid fields, never crashes |
| `build_compact_payload_dict(result)` | Builds the MQTT-ready dict |
| `build_lora_payload(result)` | Builds the compact LoRa string |
| `send_lora_with_retry(payload)` | Sends with up to 3 retries, checks ACK |
| `send_mqtt_message(payload_dict)` | Publishes to MQTT broker (paho-mqtt) |
| `process_and_send_alert(result)` | **Main entry point** — orchestrates everything |

> **Note:** `send_lora_message()` and `wait_for_lora_ack()` are placeholder implementations. Replace them with your actual serial/SPI hardware code when deploying on real LoRa hardware (e.g., SX1276, RFM95W).

---

### `elephant_bee_sound_raspberry.py` — Bee Sound Deterrent
Plays bee buzzing audio clips through a Bluetooth speaker to safely deter elephants. This is a non-harmful, wildlife-friendly deterrent method.

**Activation conditions — ALL must be true:**
| Condition | Required Value |
|-----------|---------------|
| `elephant_detected` | `True` |
| `behavior` | `"calm"` |
| `group_type` | `"herd"` or `"individual"` |
| Current time | Within allowed time windows |

**Deactivation — if ANY is true, bee sound will NOT play:**
- `elephant_detected == False`
- `behavior == "aggressive"` (never disturb an aggressive elephant)
- `group_type == "family"` (never disturb a family group)

**Why not for family groups?** Playing deterrent sounds near a family (2 adults + calves) could cause panic and separate calves from mothers, which is dangerous. The system deliberately avoids this.

**Allowed time windows (configurable):**
```python
ALLOWED_PLAY_TIME = [
    ("18:00", "23:00"),   # Evening
    ("04:30", "07:30"),   # Early morning
]
```

**Playback rules:**
- Plays for exactly **10 minutes** per activation
- Randomly selects from clips in `bee_sounds/` folder
- **Never plays the same clip twice in a row**
- Uses `mpg123` → `aplay` → `pygame` (in order of preference)

**Required audio files** — add your clips to `bee_sounds/`:
```
bee_sounds/
├── bee1.mp3
├── bee2.mp3
└── bee3.mp3
```

**Key functions:**
| Function | Purpose |
|----------|---------|
| `is_within_allowed_time()` | Checks current time against allowed windows |
| `select_bee_sound()` | Random clip selection, no consecutive repeats |
| `play_bee_sound(file, duration)` | Plays audio for specified duration |
| `should_activate_bee_sound(result)` | Evaluates all activation/deactivation conditions |
| `process_bee_sound(result)` | **Main entry point** — runs full check and plays if conditions met |

---

### `alert_system.py` — Local Alert System
Manages local hardware and network alerts:
- **GPIO**: LED indicator (pin 17) and buzzer (pin 27) on Raspberry Pi
- **Webhook**: HTTP POST to a configured URL
- **Email**: SMTP email to configured recipients

> Note: This module exists and is configured but is not currently wired into the main pipeline. LoRa/MQTT alerts in `lora_integration.py` handle remote notification.

---

### `pi_optimizer.py` — Raspberry Pi Optimization
Applies hardware-level optimizations for Raspberry Pi 5:
- Sets CPU thread count to 4 (matches Pi 5's quad-core)
- Disables GPU (Pi does not have CUDA)
- Configures OpenCV and PyTorch for efficient Pi operation

---

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 5                                    │
│                                                                      │
│  ┌──────────┐   ┌──────────────────────────────────────────────┐   │
│  │  Camera  │──▶│              main.py                         │   │
│  └──────────┘   │   (Entry point, CLI args, model checks)      │   │
│  ┌──────────┐   └────────────────────┬─────────────────────────┘   │
│  │   Mic /  │                        │                              │
│  │  Audio   │──▶ integrated_pipeline.py                            │
│  └──────────┘        │                                              │
│                       ├──▶ elephant_detector.py (YOLOv8)            │
│                       ├──▶ pose_classifier.py   (YOLOv8-pose + ML)  │
│                       └──▶ sound_classifier.py  (CNN-LSTM + RF/XGB) │
│                                      │                              │
│                              VideoResult dict                        │
│                           (behavior, group, counts)                  │
│                                      │                              │
│                    ┌─────────────────┴──────────────────┐          │
│                    │                                     │          │
│                    ▼                                     ▼          │
│         lora_integration.py               elephant_bee_sound_      │
│         ┌──────────────────┐              raspberry.py             │
│         │ Validate result  │              ┌───────────────────┐    │
│         │ Build payload    │              │ Check conditions  │    │
│         │ Try LoRa (×3)    │              │ Check time window │    │
│         │ Fallback: MQTT   │              │ Select bee clip   │    │
│         └────────┬─────────┘              │ Play 10 minutes  │    │
│                  │                        └────────┬──────────┘    │
└──────────────────┼─────────────────────────────────┼───────────────┘
                   │                                 │
        ┌──────────┼──────────┐             ┌────────▼────────┐
        │          │          │             │  Bluetooth      │
        ▼          ▼          ▼             │  Speaker        │
    LoRa TX    MQTT Broker  (both          │  (bee sounds)   │
   (primary)  (fallback)    failed:         └─────────────────┘
                             log error)
        │          │
        ▼          ▼
   Remote        MQTT
   Receiver     Subscriber
  (forest       (monitoring
   guard)        dashboard)
```

---

## Directory Structure

```
elephent_model/
├── main.py                          # Entry point
├── config.py                        # All configuration and constants
├── integrated_pipeline.py           # Core AI pipeline orchestrator
├── elephant_detector.py             # YOLOv8 elephant detection
├── pose_classifier.py               # Pose-based behavior classification
├── sound_classifier.py              # Sound-based behavior classification
├── pi_optimizer.py                  # Raspberry Pi hardware optimization
├── alert_system.py                  # Local GPIO/webhook/email alerts
├── lora_integration.py              # LoRa + MQTT wireless alert sender
├── elephant_bee_sound_raspberry.py  # Bee sound Bluetooth deterrent
├── requirements.txt                 # Python dependencies
├── setup_raspberry_pi.sh            # Automated Pi setup script
├── elephant-detection.service       # Systemd background service file
│
├── bee_sounds/                      # Bee audio clips (add your MP3s here)
│   ├── bee1.mp3
│   ├── bee2.mp3
│   └── bee3.mp3
│
├── elephent detect/
│   └── model/
│       └── best.pt                  # YOLOv8 elephant detection model
│
├── posed based/
│   └── model/
│       ├── yolov8n-pose.pt          # YOLOv8 pose estimation model
│       ├── pose_model.pkl           # Trained ML behavior classifier
│       ├── pose_scaler.pkl          # Feature scaler
│       ├── pose_label_encoder.pkl   # Label encoder
│       └── pose_feature_order.json  # Feature input order
│
├── sound-based/
│   └── model/
│       ├── cnn_lstm_model.h5        # CNN-LSTM deep learning model
│       ├── rf_model.pkl             # Random Forest model
│       ├── xgb_model.pkl           # XGBoost model
│       ├── scaler.pkl               # ML feature scaler
│       ├── label_encoder.pkl        # Label encoder
│       ├── feature_order.json       # ML feature input order
│       └── hybrid_feature_order.json
│
├── output/                          # Annotated videos and JSON results
└── logs/                            # Pipeline and alert log files
```
└── logs/                            # Pipeline and alert log files
```

---

## Installation

### Raspberry Pi 5 Quick Setup

```bash
chmod +x setup_raspberry_pi.sh
./setup_raspberry_pi.sh
source venv/bin/activate
```

### Manual Installation

**1. Install system dependencies:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv \
    libatlas-base-dev libhdf5-dev \
    libjpeg-dev libpng-dev \
    ffmpeg libportaudio2 \
    mpg123 bluetooth bluez
```

> `mpg123` is required for bee sound playback. `bluetooth`/`bluez` are needed for the Bluetooth speaker.

**2. Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Python dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Install optional wireless dependencies:**
```bash
# LoRa via MQTT fallback
pip install paho-mqtt

# Bee sound fallback player
pip install pygame
```

**5. Pair Bluetooth speaker:**
```bash
bluetoothctl
  power on
  scan on
  # Note the MAC address of your speaker
  pair XX:XX:XX:XX:XX:XX
  trust XX:XX:XX:XX:XX:XX
  connect XX:XX:XX:XX:XX:XX
  exit

# Set as default audio output
pactl list short sinks          # find your Bluetooth sink name
pactl set-default-sink <sink_name>
```

**6. Add bee sound audio clips:**
```bash
# Copy .mp3, .wav, or .ogg files into bee_sounds/
cp bee1.mp3 bee2.mp3 bee3.mp3 bee_sounds/
```

---

## Usage

### Process a Video File

```bash
# Basic detection
python main.py video.mp4

# With live preview and saved output
python main.py video.mp4 --preview --output result.mp4

# Disable sound classification (faster)
python main.py video.mp4 --no-sound

# Detection only, no behavior analysis
python main.py video.mp4 --no-sound --no-pose
```

### Live Camera Feed

```bash
python main.py --camera --preview
python main.py --camera --camera-id 0 --preview
```

### Other Commands

```bash
# Verify all model files are in place
python main.py --check-models

# Test LoRa integration independently
python lora_integration.py

# Test bee sound system independently
python elephant_bee_sound_raspberry.py
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `video` | Path to input video file |
| `--camera` | Use live camera feed |
| `--camera-id N` | Camera device ID (default: 0) |
| `--output, -o` | Path for output video |
| `--preview, -p` | Show real-time preview window |
| `--no-sound` | Skip sound-based classification |
| `--no-pose` | Skip pose-based classification |
| `--no-warmup` | Skip model warmup on startup |
| `--check-models` | Check if all model files exist |
| `--verbose, -v` | Enable verbose logging |

---

## LoRa and MQTT Wireless Alerts

### How It Works

After each video is processed, `lora_integration.py` is called with the detection result. It:

1. Validates and normalizes the result fields
2. Builds a compact LoRa payload string
3. Transmits via LoRa (primary) with retries
4. Falls back to MQTT over WiFi if LoRa fails
5. **Always transmits** — even when no elephant is detected (sends a clear-status packet)

### Payload Formats

**LoRa compact string (sent over radio):**
```
PID=PILLAR_01;LAT=7.123456;LON=80.123456;EL=1;CNT=4;ADL=3;CLF=1;GRP=herd;BEH=calm;CON=0.87;TS=2025-08-01T18:30:00
```

**No-elephant clear-status packet:**
```
PID=PILLAR_01;LAT=7.123456;LON=80.123456;EL=0;TS=2025-08-01T18:30:00
```

**MQTT JSON payload (fallback):**
```json
{
  "pillar_id": "PILLAR_01",
  "lat": 7.123456,
  "lon": 80.123456,
  "elephant": true,
  "count": 4,
  "adult": 3,
  "calf": 1,
  "group": "herd",
  "behavior": "calm",
  "confidence": 0.87,
  "ts": "2025-08-01T18:30:00"
}
```

### Configuration (`lora_integration.py`)

```python
PILLAR_ID              = "PILLAR_01"
PILLAR_LATITUDE        = 7.123456
PILLAR_LONGITUDE       = 80.123456
LORA_ACK_TIMEOUT_SECONDS = 2
LORA_MAX_RETRIES       = 3
MQTT_BROKER            = "broker.hivemq.com"
MQTT_PORT              = 1883
MQTT_TOPIC             = "elephant/alerts"
MQTT_QOS               = 1
```

### Replacing the LoRa Placeholder with Real Hardware

The `send_lora_message()` and `wait_for_lora_ack()` functions are **stubs** — replace them with your actual SX1276/RFM95W serial or SPI driver:

```python
# In send_lora_message(payload):
import serial
ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
ser.write(payload.encode('utf-8'))
ser.close()

# In wait_for_lora_ack(timeout_seconds):
import serial
ser = serial.Serial('/dev/ttyS0', 9600, timeout=timeout_seconds)
response = ser.read(1)
ser.close()
return response == b'\x06'   # ACK byte
```

### Wiring Into main.py

Paste these lines after `process_video()` returns a result:

```python
from lora_integration import process_and_send_alert
from elephant_bee_sound_raspberry import process_bee_sound
from datetime import datetime

result = pipeline.process_video(video_path)   # existing call

alert_input = {
    "elephant_detected": result.elephant_detected,
    "adult_count":        result.max_adult_count,
    "calf_count":         result.max_calf_count,
    "elephant_count":     result.max_adult_count + result.max_calf_count,
    "group_type":         result.dominant_group_classification,
    "behavior":           result.dominant_behavior,
    "confidence":         result.dominant_behavior_confidence,
    "timestamp":          datetime.now().isoformat(),
}

process_and_send_alert(alert_input)
process_bee_sound(alert_input)
```

---

## Bee Sound Deterrent

### Science Behind It

Elephants are naturally averse to the sound of bees. Playing bee buzzing audio through a Bluetooth speaker near a herd or lone elephant (in a calm state) causes them to retreat without harm — a proven, non-violent HWC mitigation strategy.

### Activation Conditions

All three conditions must be true simultaneously:

| Condition | Required Value |
|-----------|----------------|
| `elephant_detected` | `True` |
| `behavior` | `"calm"` |
| `group_type` | `"herd"` or `"individual"` |
| Time of day | Within allowed window |

**Never activates for:**

| Scenario | Reason |
|----------|--------|
| `behavior = "aggressive"` | Loud sound may escalate aggression |
| `group_type = "family"` | Calves present — distress could trigger charge |
| `elephant_detected = False` | No target |
| Outside time window | Reduces false activations at low-risk hours |

### Time Windows

```python
ALLOWED_PLAY_TIME = [
    ("18:00", "23:00"),   # Evening — elephants leave forest for farms
    ("04:30", "07:30"),   # Early morning — return movement
]
```

Edit `ALLOWED_PLAY_TIME` in `elephant_bee_sound_raspberry.py` to match your local elephant activity patterns.

### Audio Clip Setup

Place `.mp3`, `.wav`, or `.ogg` files in the `bee_sounds/` directory. The system:
- Randomly selects a clip each activation
- Never plays the same clip twice in a row (if more than one clip is available)
- Loops the selected clip for `PLAYBACK_DURATION_MINUTES` (default: 10 minutes)
- Tries `mpg123` → `aplay` → `pygame` in order of preference

```bash
bee_sounds/
├── bee1.mp3    # source: royalty-free bee audio
├── bee2.mp3
└── bee3.wav
```

### Return Value of `process_bee_sound()`

```python
{
    "activated": True,
    "reason": "conditions_met_playing",
    "clip": "bee_sounds/bee2.mp3",
    "played_minutes": 10
}
```

Possible `reason` values: `"conditions_met_playing"`, `"conditions_met_but_outside_time_window"`, `"no_elephant"`, `"aggressive_behavior"`, `"family_group_detected"`, `"no_audio_clips_found"`.

---

## Output Format

### Console Output

```
==========================================================
ELEPHANT DETECTION & BEHAVIOR CLASSIFICATION RESULTS
==========================================================
Video: trail_cam_2025.mp4
Duration: 30.00s | Frames: 900
Processing Time: 47.12s
----------------------------------------------------------
Elephant Detected:       YES
Max Adults:              3  |  Max Calves: 1
Group Classification:    HERD
Dominant Behavior:       CALM  (confidence: 0.87)
Behavior Source:         pose
==========================================================
LoRa alert sent:  PILLAR_01 → PID=PILLAR_01;LAT=7.123456;...
Bee sound:        ACTIVATED — bee_sounds/bee1.mp3 (10 min)
==========================================================
```

### JSON Results

Saved to `output/results_<video_name>.json`:

```json
{
  "video_path": "trail_cam_2025.mp4",
  "duration": 30.0,
  "elephant_detected": true,
  "dominant_group_classification": "herd",
  "dominant_behavior": "calm",
  "dominant_behavior_confidence": 0.87,
  "dominant_behavior_source": "pose",
  "max_adult_count": 3,
  "max_calf_count": 1,
  "frame_results": [{ "..." }]
}
```

---

## Running as a Service

To continuously process camera frames in the background:

```bash
# Copy and configure the service file
sudo cp elephant-detection.service /etc/systemd/system/
sudo nano /etc/systemd/system/elephant-detection.service
# Update WorkingDirectory and ExecStart paths

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable elephant-detection
sudo systemctl start elephant-detection

# Monitor
sudo systemctl status elephant-detection
sudo journalctl -u elephant-detection -f
```

---

## Alert Configuration

### LoRa / MQTT (lora_integration.py)

| Constant | Default | Description |
|----------|---------|-------------|
| `PILLAR_ID` | `"PILLAR_01"` | Unique ID of this sensor pillar |
| `PILLAR_LATITUDE` | `7.123456` | GPS latitude of the pillar |
| `PILLAR_LONGITUDE` | `80.123456` | GPS longitude of the pillar |
| `LORA_ACK_TIMEOUT_SECONDS` | `2` | Seconds to wait for LoRa ACK |
| `LORA_MAX_RETRIES` | `3` | Retry attempts before MQTT fallback |
| `MQTT_BROKER` | `"broker.hivemq.com"` | MQTT broker hostname |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `"elephant/alerts"` | MQTT publish topic |
| `MQTT_QOS` | `1` | MQTT Quality of Service level |

### Bee Sound (elephant_bee_sound_raspberry.py)

| Constant | Default | Description |
|----------|---------|-------------|
| `BEE_SOUNDS_DIR` | `"bee_sounds"` | Directory containing audio clips |
| `PLAYBACK_DURATION_MINUTES` | `10` | How long to play each activation |
| `ALLOWED_PLAY_TIME` | See above | Active time windows (list of tuples) |

### GPIO / Email / Webhook (alert_system.py)

```python
DEFAULT_ALERT_CONFIG = {
    'cooldown_seconds': 30,
    'gpio_enabled':     True,
    'gpio_led_pin':     17,
    'gpio_buzzer_pin':  27,
    'webhook_url':      'https://your-endpoint/alert',
    'email_enabled':    True,
    'smtp_server':      'smtp.gmail.com',
    'smtp_port':        587,
    'smtp_username':    'your@email.com',
    'smtp_password':    'app_password',
    'email_recipients': ['ranger@park.org'],
}
```

---

## Performance Optimization

The pipeline is tuned for Raspberry Pi 5:

| Technique | Detail |
|-----------|--------|
| Frame skipping | Processes every 5th frame (configurable) |
| Resolution capping | Scales to 640×480 before inference |
| Thread limit | 4 threads matching Pi 5 core count |
| Model warmup | Loads and pre-runs models before first real frame |
| Temporal smoothing | Behavior averaged over 5 consecutive frames |
| Sound conditional | Sound classification skipped if no elephant detected in frame |

### Tuning in config.py

```python
FRAME_SKIP       = 5          # Higher = faster, less accurate
MAX_RESOLUTION   = (640, 480) # Lower = faster inference
NUM_THREADS      = 4          # Match your CPU core count
SMOOTHING_WINDOW = 5          # Frames for behavior averaging
```

---

## Model Information

### Elephant Detection — YOLOv8 (custom)

| Item | Detail |
|------|--------|
| Architecture | YOLOv8 |
| Classes | Adult elephant, Calf |
| Input size | 640×640 |
| File | `elephent detect/model/best.pt` |

### Pose Classification — YOLOv8-pose + Random Forest

| Item | Detail |
|------|--------|
| Pose backbone | YOLOv8-pose (`yolov8n-pose.pt`) |
| Classifier | Rule-based + Random Forest |
| Feature count | 17 pose-based features |
| Files | `posed based/model/` |

### Sound Classification — CNN-LSTM + Ensemble

| Item | Detail |
|------|--------|
| Deep learning | CNN-LSTM (`.h5`) |
| ML ensemble | Random Forest + XGBoost |
| Feature count | 62 (MFCCs, spectral centroid, ZCR, chroma, …) |
| Files | `sound-based/model/` |

---

## Troubleshooting

### Video / Camera

| Problem | Fix |
|---------|-----|
| `"Could not open video"` | Check file path and format (MP4, AVI, MKV supported) |
| `"Camera not opening"` | Try `--camera-id 1`, check permissions with `ls -l /dev/video*` |
| `"Out of memory"` | Increase `FRAME_SKIP`, lower `MAX_RESOLUTION` in config.py |
| TensorFlow errors | Install ARM build: `pip install tensorflow-aarch64` |
| Missing model files | Run `python main.py --check-models` |

### LoRa / MQTT

| Problem | Fix |
|---------|-----|
| LoRa always falls back to MQTT | Replace stub functions in `send_lora_message()` and `wait_for_lora_ack()` with real serial/SPI driver |
| `paho-mqtt not found` | `pip install paho-mqtt` |
| MQTT connection refused | Check `MQTT_BROKER` and `MQTT_PORT`; verify internet on Pi |
| All retries fail, MQTT also fails | Check `lora_integration.log` for detailed error trace |

### Bee Sound

| Problem | Fix |
|---------|-----|
| No audio plays | Check `bee_sounds/` directory is not empty; add `.mp3` files |
| `mpg123: command not found` | `sudo apt-get install mpg123` |
| Sound plays from Pi speaker not Bluetooth | Run `pactl set-default-sink <bt_sink>` and reconnect speaker |
| Bluetooth not pairing | Run `bluetoothctl` interactively; ensure speaker is in pairing mode |
| Outside time window all the time | Check system clock (`date`); adjust `ALLOWED_PLAY_TIME` |

### General

| Problem | Fix |
|---------|-----|
| Missing logs | Check `logs/` directory; ensure write permissions |
| Behavior always "calm" | Inspect `pose_rules.json` thresholds; verify pose model is loaded |
| Sound classification ignored | Confirm `.h5` and `feature_order.json` are in `sound-based/model/` |

---

## Contributors

- Tharindu M Patipolaarachchi - Project Owner and Primary Developer
- Full list and contributor guidelines: [CONTRIBUTORS.md](CONTRIBUTORS.md)

---

## License

This project is for educational and conservation research purposes.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- TensorFlow / Keras by Google
- OpenCV community
- HiveMQ public MQTT broker (testing only — use a private broker in production)
