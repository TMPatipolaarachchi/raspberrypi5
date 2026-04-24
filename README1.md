# README1 - Full Project Understanding and Website Development Blueprint

## 1. Purpose of This Document
This document gives a complete, practical understanding of the Elephant Detection project and translates it into a website implementation plan.

Goal:
- Understand the whole codebase, models, data flow, deployment, and integrations.
- Use this as the source plan to build a production-ready website/dashboard.

Important note about Website.pdf:
- The repository contains Website.pdf, but this environment cannot extract PDF text because PDF parsing tools are unavailable.
- This README1 is built from the full repository code and docs, and is still complete enough to drive website development.

---

## 2. Project Mission
This system is an AI-enabled wildlife safety solution for human-elephant conflict mitigation.

Core mission:
- Detect elephants from video/camera.
- Classify elephant behavior: aggressive or calm.
- Classify group type: individual, family, herd.
- Send alert data from Raspberry Pi to ESP32 over serial (current implementation).
- Activate bee-sound deterrent only in safe conditions.

Main operating environment:
- Raspberry Pi 5 edge device.
- Camera and optional microphone/audio input.
- Local model inference with YOLO + ML + DL.

---

## 3. End-to-End Runtime Flow
1. Input comes from either:
- Video file path, or
- Live camera stream.

2. Frame processing pipeline:
- Elephant detection model finds adults/calves and counts.
- Pose model predicts behavior from body keypoints.
- Sound model predicts behavior from extracted audio.

3. Fusion and smoothing:
- Pose + sound outputs are fused.
- Temporal smoothing stabilizes behavior output.
- Final output is guaranteed to be one of:
  - aggressive
  - calm

4. Final decision output includes:
- elephant_detected
- adult_count
- calf_count
- elephant_count
- group_type
- behavior
- confidence
- timestamp

5. Action modules:
- ESP32 serial message sender sends payload.
- Bee sound module decides whether to activate deterrent.

---

## 4. Repository Structure and What Each Part Does

### 4.1 Core application files
- main.py
  - Main CLI entry point.
  - Handles args, model checks, environment setup, and final result printing.
  - Calls integrated pipeline, ESP32 sender, and bee sound module.

- integrated_pipeline.py
  - Central orchestrator of detection and behavior analysis.
  - Handles frame skipping, optional pose/sound classifiers, behavior fusion and smoothing.
  - Produces VideoResult summary and optional JSON outputs.

- config.py
  - Central constants and model paths.
  - Includes standardized behavior label mapping and normalization logic.

### 4.2 AI module files
- elephant_detector.py
  - YOLOv8 detector for elephant classes (adult/calf).
  - Determines group classification based on counts.

- pose_classifier.py
  - YOLOv8 pose keypoint extraction + ML classifier.
  - Computes engineered pose features and predicts behavior.

- sound_classifier.py
  - CNN-LSTM + RF + XGB ensemble for audio behavior classification.
  - Includes robust error handling and feature extraction pipeline.

- diagnose_sound_models.py
  - Diagnostic script for model outputs, class labels, and shape consistency.

### 4.3 Alerting and deterrent files
- esp32_serial_integration.py
  - Current alert transport implementation.
  - Validates result, normalizes values, builds payload, and writes to serial UART.
  - Sends full payload when elephant detected, minimal payload when not detected.

- elephant_bee_sound_raspberry.py
  - Bee sound deterrent logic.
  - Activation only when all conditions are safe.

- alert_system.py
  - General local alert manager (GPIO, webhook, email).
  - Appears available but not the active path used in main runtime flow.

### 4.4 Edge/camera/performance files
- raspberry_pi_camera_input.py
  - Camera module abstraction with picamera2 first and OpenCV fallback.
  - Stream handling, frame generator, and self-test.

- pi_optimizer.py
  - Runtime optimizations for Pi, thread and dependency checks, warmup utilities.

### 4.5 Deployment and setup files
- setup_raspberry_pi.sh
  - Automated Raspberry Pi package and virtualenv setup.

- elephant-detection.service
  - Systemd service file for background startup.

- requirements.txt
  - Main Python dependencies.

- posed based/requirements.txt
- sound-based/requirements.txt
  - Submodule dependency definitions.

### 4.6 Documentation and status files
- README.md
  - Existing full project documentation and architecture details.

- CHANGES_BY_FILE.md
- DEBUGGING_REPORT.md
- FIX_SUMMARY.md
  - Notes of recent fixes and label normalization improvements.

### 4.7 Model and data folders
- elephent detect/model/best.pt
  - Elephant detector model weights.

- posed based/model/
  - Pose-related model files and metadata.
  - Includes feature order, importances, rules, and training metadata JSON.

- sound-based/model/
  - Audio models and scalers/encoders.
  - Includes feature order JSON files.

- bee_sounds/
  - Audio clips for deterrent playback.

- logs/
  - Runtime logs.

### 4.8 Other artifacts
- video.mp4
  - Sample video artifact.

- Website.pdf
  - Website-related external document in repo.

- __pycache__/
  - Python bytecode cache.

---

## 5. Verified Business Rules in Code

### Behavior normalization
- Behavior labels are normalized to exactly:
  - aggressive
  - calm
- Any unknown/missing value defaults safely to calm.

### Group classification
- individual: one elephant.
- family: 2 adults + at least 1 calf.
- herd: other multi-elephant combinations.

### Bee sound safety rules
Bee sound should activate only if:
- elephant_detected is true
- behavior is calm
- group_type is herd or individual
- current time is within allowed windows

Bee sound should never activate if:
- no elephant detected
- behavior is aggressive
- group is family

### Alert payload behavior
- elephant detected: full payload with counts, group, behavior, confidence, timestamp.
- no elephant: minimal payload still sent as device status heartbeat.

---

## 6. What the Website Should Achieve
The website should be the control + monitoring interface for this edge AI system.

Primary objectives:
1. Real-time monitoring of pillar status.
2. Real-time alerts and event timeline.
3. Historical analytics and filtering.
4. Device/system health monitoring.
5. Configuration management (safe, role-protected).
6. Optional remote operations (test alert, test bee sound, restart service).

---

## 7. Recommended Website Information Architecture

### Page 1: Live Dashboard
Show:
- Current status card: elephant detected yes/no.
- Behavior status: aggressive/calm with confidence.
- Group type and counts.
- Last update timestamp.
- Alert transport status (serial sent / failed).
- Bee sound state.

### Page 2: Event Timeline
Show:
- Chronological list of all detections and no-detection heartbeats.
- Filters by behavior, group type, date range, severity.
- Event detail modal with full payload JSON.

### Page 3: Map View
Show:
- Pillar location markers.
- Marker color by latest risk level.
- Click marker for current state and last events.

### Page 4: Media Review
Show:
- Annotated output clips/images (if saved).
- Event-linked media playback.

### Page 5: Device Health
Show:
- CPU/memory/temp (if exposed).
- Service state and uptime.
- Last successful model run.
- Error logs summary.

### Page 6: Configuration Center
Editable settings with role protection:
- Frame skip.
- Confidence thresholds.
- Allowed bee sound time windows.
- Alert cooldown.
- Serial settings.

### Page 7: Reports and Export
- Daily/weekly/monthly reports.
- Export CSV/JSON/PDF of events.
- Trend charts by behavior and group type.

---

## 8. Data Model for the Website Backend
Use this as a starting schema.

### Table: pillars
- id
- pillar_id
- lat
- lon
- name
- is_active
- created_at

### Table: detections
- id
- pillar_id
- elephant_detected
- adult_count
- calf_count
- elephant_count
- group_type
- behavior
- confidence
- behavior_source
- processing_time
- timestamp
- raw_payload_json

### Table: deterrent_actions
- id
- pillar_id
- activated
- reason
- clip
- played_minutes
- timestamp

### Table: alert_transmissions
- id
- pillar_id
- sent
- transport
- reason
- payload_json
- timestamp

### Table: system_health
- id
- pillar_id
- cpu_percent
- memory_percent
- temp_c
- service_status
- timestamp

### Table: users
- id
- name
- email
- role (admin, ranger, viewer)
- password_hash
- created_at

---

## 9. API Contract Suggestion

### Ingestion APIs
- POST /api/v1/ingest/detection
  - Receives final pipeline output.

- POST /api/v1/ingest/alert-status
  - Receives serial send outcome.

- POST /api/v1/ingest/bee-status
  - Receives deterrent activation outcome.

### Query APIs
- GET /api/v1/dashboard/current
- GET /api/v1/events
- GET /api/v1/events/{id}
- GET /api/v1/pillars
- GET /api/v1/health/latest
- GET /api/v1/reports/summary

### Control APIs (admin only)
- POST /api/v1/control/test-alert
- POST /api/v1/control/test-bee-sound
- POST /api/v1/control/restart-service
- PATCH /api/v1/config

### Realtime channel
- WebSocket /ws/live
  - Pushes latest detection and alert events.

---

## 10. Frontend Stack Recommendation
- Framework: Next.js or React + Vite.
- UI toolkit: Tailwind CSS + charting library.
- Maps: Leaflet or Mapbox.
- Real-time: WebSocket client.
- State: React Query + lightweight global store.

Design recommendations:
- Use clear color semantics:
  - Red for aggressive.
  - Green for calm/no risk.
  - Amber for warning conditions.
- Keep critical KPIs always visible in top section.
- Build mobile-first ranger view with large tap targets.

---

## 11. Backend Stack Recommendation
- Python FastAPI (good alignment with existing Python code), or Node.js if preferred.
- PostgreSQL for events and analytics.
- Redis optional for caching and pub/sub.
- Background worker for report generation and retention cleanup.

Edge integration options:
- Option A: Raspberry Pi pushes events to web backend.
- Option B: Web backend pulls logs/results from edge nodes.
- Recommended: Option A push model for near real-time dashboard updates.

---

## 12. Security and Reliability Requirements
- JWT auth with role-based access.
- TLS for all API and WebSocket traffic.
- Signed device keys for ingest endpoints.
- Rate limiting and replay protection on ingest.
- Audit logs for config changes and control actions.
- Offline queueing on edge node when connectivity drops.

---

## 13. Website Features Mapped to Existing Code

### Detection summary cards
Source fields from pipeline result:
- elephant_detected
- max_adult_count
- max_calf_count
- dominant_group_classification
- dominant_behavior
- dominant_behavior_confidence

### Alert status panel
Source from ESP32 sender status:
- sent
- transport
- reason
- payload

### Bee deterrent panel
Source from bee module status:
- activated
- reason
- clip
- played_minutes

### System health panel
Source candidates:
- pi_optimizer system info
- service status and logs

---

## 14. Implementation Phases

### Phase 1 - Foundation
- Build backend ingest endpoint.
- Store detection and alert records.
- Build basic live dashboard page.

### Phase 2 - Realtime and History
- Add WebSocket live updates.
- Add timeline/events list with filters.
- Add detailed event view.

### Phase 3 - Device Ops and Config
- Add health page and service status checks.
- Add admin config UI and protected control APIs.

### Phase 4 - Reporting and Mapping
- Add map page and geospatial visualization.
- Add exports and scheduled reports.

### Phase 5 - Hardening
- Add auth roles, audit logs, device auth keys.
- Add retry, buffering, monitoring, and alerting for backend.

---

## 15. Immediate Development Checklist
1. Create backend project with ingest + query endpoints.
2. Define DB migrations for detections/alerts/deterrent tables.
3. Add Raspberry Pi client sender to call ingest endpoints after each run.
4. Build Dashboard page using current output fields.
5. Build Timeline and Event Detail pages.
6. Add WebSocket live updates.
7. Add health and config pages.
8. Add role-based auth.
9. Add map and report pages.
10. Run pilot with one pillar, then scale to multiple pillars.

---

## 16. Known Gaps and Recommendations
- Website.pdf content could not be parsed in this environment; review it manually and merge any extra UI/business requirements into this README1.
- Some docs mention LoRa/MQTT flow, while current executable path uses ESP32 serial. Choose one canonical production transport and reflect that in website labels.
- alert_system.py is rich but not fully wired into main run path; decide whether to integrate it in final architecture.
- Add persistent event storage from day one; current scripts are mostly runtime and logs based.

---

## 17. Final Practical Summary
You already have a strong edge AI pipeline. The website should now become the command center that does four things well:
- Observe: live state and current risk.
- Understand: event history, media, trends, reports.
- Act: safe remote controls and clear operations.
- Trust: secure ingest, reliable delivery, auditable changes.

This README1 is designed to be the build blueprint for that website.