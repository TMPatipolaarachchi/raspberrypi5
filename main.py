#!/usr/bin/env python3
"""
Elephant Detection and Behavior Classification System
Main Application Entry Point for Raspberry Pi 5

This application processes video input to:
1. Detect elephants and classify them as adults or calves
2. Count elephants and determine group classification (individual, family, herd)
3. Classify behavior as aggressive or calm using pose and sound analysis

Usage:
    python main.py <video_path> [options]
    python main.py --camera [options]
    python main.py --help

Examples:
    python main.py video.mp4
    python main.py video.mp4 --preview --output result.mp4
    python main.py --camera --preview
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment for Raspberry Pi optimization"""
    # Suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Import and apply optimizations
    try:
        from pi_optimizer import RaspberryPiOptimizer
        optimizer = RaspberryPiOptimizer()
        system_info = optimizer.apply_all_optimizations()
        
        logger.info("System Information:")
        for key, value in system_info.items():
            logger.info(f"  {key}: {value}")
        
        return optimizer
    except Exception as e:
        logger.warning(f"Could not apply optimizations: {e}")
        return None


def check_models():
    """Check if all required model files exist"""
    from config import (
        ELEPHANT_DETECT_MODEL, POSE_YOLO_MODEL, POSE_CLASSIFIER,
        POSE_SCALER, POSE_LABEL_ENCODER, SOUND_CNN_LSTM_MODEL,
        SOUND_RF_MODEL, SOUND_XGB_MODEL, SOUND_SCALER, SOUND_LABEL_ENCODER
    )
    
    models = {
        'Elephant Detection': ELEPHANT_DETECT_MODEL,
        'Pose YOLO': POSE_YOLO_MODEL,
        'Pose Classifier': POSE_CLASSIFIER,
        'Pose Scaler': POSE_SCALER,
        'Pose Label Encoder': POSE_LABEL_ENCODER,
        'Sound CNN-LSTM': SOUND_CNN_LSTM_MODEL,
        'Sound RF Model': SOUND_RF_MODEL,
        'Sound XGB Model': SOUND_XGB_MODEL,
        'Sound Scaler': SOUND_SCALER,
        'Sound Label Encoder': SOUND_LABEL_ENCODER
    }
    
    missing = []
    for name, path in models.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
            logger.error(f"Missing model: {name} at {path}")
        else:
            logger.info(f"Found model: {name}")
    
    return len(missing) == 0, missing


def process_video(args):
    """Process video through the pipeline"""
    from integrated_pipeline import IntegratedPipeline
    from pi_optimizer import warmup_models
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = IntegratedPipeline(
        enable_sound=not args.no_sound,
        enable_pose=not args.no_pose
    )
    
    # Warmup models
    if not args.no_warmup:
        warmup_models(pipeline)
    
    # Process video
    result = pipeline.process_video(
        args.video,
        output_path=args.output,
        show_preview=args.preview
    )
    
    return result


def process_camera(args):
    """Process live camera feed"""
    import cv2
    from integrated_pipeline import IntegratedPipeline
    from pi_optimizer import warmup_models, FrameRateController
    from config import MAX_RESOLUTION
    
    # Initialize pipeline
    logger.info("Initializing pipeline for camera...")
    pipeline = IntegratedPipeline(
        enable_sound=not args.no_sound,
        enable_pose=not args.no_pose
    )
    
    # Warmup
    if not args.no_warmup:
        warmup_models(pipeline)
    
    # Open camera
    camera_id = args.camera_id if hasattr(args, 'camera_id') else 0
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"Could not open camera {camera_id}")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    logger.info("Starting camera processing (press 'q' to quit)...")
    
    # Frame rate controller
    fps_controller = FrameRateController(target_fps=10)
    
    # Sound result (updated periodically)
    sound_result = None
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % 3 != 0:
                continue
            
            # Process frame
            detection = pipeline.elephant_detector.detect(frame, annotate=False)
            
            # Pose classification
            pose_result = None
            if pipeline.pose_classifier and detection.elephant_detected:
                pose_result = pipeline.pose_classifier.classify(frame)
            
            # Combine behavior
            behavior = pipeline._combine_behavior_results(pose_result, sound_result)
            
            # Annotate and display
            annotated = pipeline._annotate_frame(frame, detection, behavior)
            
            cv2.imshow("Elephant Detection - Live", annotated)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                cv2.imwrite(f"screenshot_{frame_count}.jpg", annotated)
                logger.info(f"Screenshot saved: screenshot_{frame_count}.jpg")
            
            # Control frame rate
            fps_controller.wait()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Elephant Detection and Behavior Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a video file:
    python main.py video.mp4

  Process with preview and save output:
    python main.py video.mp4 --preview --output result.mp4

  Live camera processing:
    python main.py --camera --preview

  Disable sound classification (faster):
    python main.py video.mp4 --no-sound
        """
    )
    
    # Input options
    parser.add_argument('video', type=str, nargs='?', help='Path to input video file')
    parser.add_argument('--camera', action='store_true', help='Use live camera feed')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID (default: 0)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, help='Path for output video')
    parser.add_argument('--preview', '-p', action='store_true', help='Show real-time preview')
    
    # Processing options
    parser.add_argument('--no-sound', action='store_true', help='Disable sound-based classification')
    parser.add_argument('--no-pose', action='store_true', help='Disable pose-based classification')
    parser.add_argument('--no-warmup', action='store_true', help='Skip model warmup')
    
    # System options
    parser.add_argument('--check-models', action='store_true', help='Check if all model files exist')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup environment
    setup_environment()
    
    # Check models if requested
    if args.check_models:
        success, missing = check_models()
        if success:
            print("\n✓ All model files found!")
        else:
            print("\n✗ Missing model files:")
            for m in missing:
                print(f"  - {m}")
        return 0 if success else 1
    
    # Validate input
    if not args.video and not args.camera:
        parser.error("Please provide either a video file or use --camera for live feed")
    
    if args.video and not os.path.exists(args.video):
        parser.error(f"Video file not found: {args.video}")
    
    # Check models
    success, missing = check_models()
    if not success:
        logger.error("Missing model files. Run with --check-models for details.")
        return 1
    
    try:
        if args.camera:
            process_camera(args)
        else:
            result = process_video(args)
            
            # Print final summary — behavior is always "aggressive" or "calm" (lowercase)
            # .upper() renders as AGGRESSIVE / CALM in the output
            if result:
                # ── Build shared input dict for alert + bee sound ──────
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

                # ── ESP32 serial alert ────────────────────────────────
                try:
                    from esp32_serial_integration import process_and_send_to_esp32
                    alert_status = process_and_send_to_esp32(alert_input)
                except Exception as e:
                    logger.warning(f"Alert send failed: {e}")
                    alert_status = {"sent": False, "error": str(e)}

                # ── Bee sound deterrent ───────────────────────────────
                try:
                    from elephant_bee_sound_raspberry import process_bee_sound
                    bee_status = process_bee_sound(alert_input)
                except Exception as e:
                    logger.warning(f"Bee sound failed: {e}")
                    bee_status = {"activated": False, "reason": str(e), "clip": None}

                # ── Console summary ───────────────────────────────────
                print("\n" + "=" * 48)
                print("FINAL ANALYSIS RESULTS")
                print("=" * 48)
                print(f"Elephant Detected: {'YES' if result.elephant_detected else 'NO'}")
                if result.elephant_detected:
                    print(f"Group Type:        {result.dominant_group_classification.upper()}")
                    print(f"Max Adults:        {result.max_adult_count}")
                    print(f"Max Calves:        {result.max_calf_count}")
                    print(f"Behavior:          {result.dominant_behavior.upper()}")
                    print(f"Behavior Source:   {result.dominant_behavior_source.upper()}")
                    print(f"Confidence:        {result.dominant_behavior_confidence:.2f}")
                print(f"Processing Time:   {result.processing_time:.2f} seconds")
                if alert_status.get("sent"):
                    transport = alert_status.get("transport", "unknown").upper()
                    print(f"Alert Sent:        YES  ({transport})")
                else:
                    err = alert_status.get("error", "unknown error")
                    print(f"Alert Sent:        NO   ({err})")
                if bee_status.get("activated"):
                    clip = os.path.basename(bee_status.get("clip") or "")
                    mins = bee_status.get("played_minutes", 0)
                    print(f"Bee Sound:         ACTIVATED  — {clip} ({mins} min)")
                else:
                    reason = bee_status.get("reason", "")
                    print(f"Bee Sound:         OFF  ({reason})")
                print("=" * 48)
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 0
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
