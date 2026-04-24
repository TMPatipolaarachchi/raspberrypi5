"""
Integrated Elephant Detection and Behavior Classification Pipeline
Combines elephant detection, pose-based, and sound-based classification
Optimized for Raspberry Pi 5
"""

import cv2
import numpy as np
import json
import os
import sys
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

from config import (
    FRAME_SKIP,
    MAX_RESOLUTION,
    BEHAVIOR_CONFIDENCE_THRESHOLD,
    BEHAVIOR_SMOOTHING_WINDOW,
    OUTPUT_DIR,
    LOG_DIR,
    SAVE_ANNOTATED_VIDEO,
    SAVE_JSON_RESULTS,
    normalize_behavior_label,   # FIX: shared label normalizer
)

from elephant_detector import ElephantDetector, DetectionResult
from pose_classifier import PoseBehaviorClassifier, PoseResult
from sound_classifier import SoundBehaviorClassifier, SoundResult, extract_audio_from_video

# Configure logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BehaviorResult:
    """Combined behavior classification result."""
    behavior: str       # ONLY "aggressive" or "calm" — never "unknown" or any other value
    confidence: float
    pose_behavior: str
    pose_confidence: float
    sound_behavior: str
    sound_confidence: float
    # Tracks which branch(es) produced the final answer (internal diagnostic only)
    behavior_source: str = "none"  # "fused", "pose_only", "sound_only", "none"


@dataclass 
class FrameResult:
    """Complete result for a single frame."""
    frame_number: int
    timestamp: float
    elephant_detected: bool
    adult_count: int
    calf_count: int
    total_count: int
    group_classification: str
    behavior: str
    behavior_confidence: float
    pose_behavior: str
    pose_confidence: float
    sound_behavior: str
    sound_confidence: float
    behavior_source: str = "none"   # propagated from BehaviorResult


@dataclass
class VideoResult:
    """Complete result for a video."""
    video_path: str
    duration: float
    total_frames: int
    processed_frames: int
    fps: float
    resolution: Tuple[int, int]
    elephant_detected: bool
    dominant_group_classification: str
    dominant_behavior: str
    max_adult_count: int
    max_calf_count: int
    frame_results: List[FrameResult]
    processing_time: float
    # dominant_behavior_source and dominant_behavior_confidence for the dominant behavior
    dominant_behavior_source: str = "none"
    dominant_behavior_confidence: float = 0.0


class IntegratedPipeline:
    """
    Main integrated pipeline for elephant detection and behavior classification
    Processes video input through all models and combines outputs
    """
    
    def __init__(self, enable_sound: bool = True, enable_pose: bool = True):
        """
        Initialize the integrated pipeline
        
        Args:
            enable_sound: Enable sound-based classification
            enable_pose: Enable pose-based classification
        """
        logger.info("Initializing Integrated Pipeline...")
        
        self.enable_sound = enable_sound
        self.enable_pose = enable_pose
        
        # Initialize models
        logger.info("Loading elephant detector...")
        self.elephant_detector = ElephantDetector()
        
        if enable_pose:
            logger.info("Loading pose classifier...")
            self.pose_classifier = PoseBehaviorClassifier()
        else:
            self.pose_classifier = None
        
        if enable_sound:
            logger.info("Loading sound classifier...")
            self.sound_classifier = SoundBehaviorClassifier()
        else:
            self.sound_classifier = None
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Behavior smoothing buffer
        self.behavior_buffer = []
        
        logger.info("Integrated Pipeline initialized successfully")
    
    def process_video(self, video_path: str, output_path: str = None,
                     show_preview: bool = False) -> VideoResult:
        """
        Process a video through the complete pipeline
        
        Args:
            video_path: Path to input video
            output_path: Path for output annotated video (optional)
            show_preview: Show real-time preview
            
        Returns:
            VideoResult with all detection and classification results
        """
        start_time = time.time()
        video_path = str(video_path)
        
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
        
        # Process audio in parallel if sound classification is enabled
        sound_result = None
        if self.enable_sound:
            logger.info("Extracting and processing audio...")
            sound_result = self._process_audio_async(video_path)
        
        # Setup video writer if output requested
        video_writer = None
        if output_path or SAVE_ANNOTATED_VIDEO:
            if output_path is None:
                output_path = str(OUTPUT_DIR / f"annotated_{Path(video_path).name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_results = []
        frame_count = 0
        processed_count = 0
        
        # Reset behavior buffer
        self.behavior_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Skip frames for efficiency
            if frame_count % FRAME_SKIP != 0:
                if video_writer:
                    video_writer.write(frame)
                continue
            
            processed_count += 1
            
            # Resize frame if needed for Pi processing
            process_frame = self._resize_frame(frame)
            
            # Run elephant detection
            detection_result = self.elephant_detector.detect(process_frame, annotate=False)
            
            # Run pose classification if elephant detected and enabled
            pose_result = None
            if self.enable_pose and detection_result.elephant_detected:
                pose_result = self.pose_classifier.classify(process_frame)
            
            # Combine behavior results
            behavior_result = self._combine_behavior_results(pose_result, sound_result)
            
            # Apply temporal smoothing
            smoothed_behavior = self._smooth_behavior(behavior_result)
            
            # Create frame result
            frame_result = FrameResult(
                frame_number=frame_count,
                timestamp=timestamp,
                elephant_detected=detection_result.elephant_detected,
                adult_count=detection_result.adult_count,
                calf_count=detection_result.calf_count,
                total_count=detection_result.total_count,
                group_classification=detection_result.group_classification,
                behavior=smoothed_behavior.behavior,
                behavior_confidence=smoothed_behavior.confidence,
                pose_behavior=smoothed_behavior.pose_behavior,
                pose_confidence=smoothed_behavior.pose_confidence,
                sound_behavior=smoothed_behavior.sound_behavior,
                sound_confidence=smoothed_behavior.sound_confidence,
                behavior_source=smoothed_behavior.behavior_source,   # FIX: propagate source
            )
            frame_results.append(frame_result)
            
            # Annotate frame
            annotated = self._annotate_frame(frame, detection_result, smoothed_behavior)
            
            # Write to output video
            if video_writer:
                video_writer.write(annotated)
            
            # Show preview
            if show_preview:
                cv2.imshow("Elephant Detection & Behavior Classification", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Log progress
            if processed_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        
        # Calculate aggregate results
        video_result = self._create_video_result(
            video_path, duration, total_frames, processed_count,
            fps, (width, height), frame_results, processing_time
        )
        
        # Save JSON results
        if SAVE_JSON_RESULTS:
            self._save_results(video_path, video_result)
        
        logger.info(f"Processing complete in {processing_time:.2f}s")
        self._print_summary(video_result)
        
        return video_result
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for efficient processing on Raspberry Pi"""
        h, w = frame.shape[:2]
        max_w, max_h = MAX_RESOLUTION
        
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(frame, (new_w, new_h))
        return frame
    
    def _process_audio_async(self, video_path: str) -> Optional[SoundResult]:
        """Extract and process audio from video"""
        try:
            # Extract audio to temp file
            audio_path = extract_audio_from_video(video_path)
            
            if audio_path is None:
                logger.warning("Could not extract audio from video")
                return None
            
            # Classify audio
            result = self.sound_classifier.classify_from_file(audio_path)
            
            # Clean up temp file
            try:
                os.remove(audio_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None
    
    def _combine_behavior_results(self, pose_result: Optional[PoseResult],
                                  sound_result: Optional[SoundResult]) -> BehaviorResult:
        """
        Combine pose and sound behavior results into a single BehaviorResult.

        Fusion rules (safe fallback at every level):
        - Both valid  -> weighted combination (pose 0.5, sound 0.5); source = "fused"
        - Pose only   -> use pose result;  source = "pose_only"
        - Sound only  -> use sound result; source = "sound_only"
        - Neither     -> behavior = "calm", confidence = 0.0; source = "none"

        All incoming labels are normalized via normalize_behavior_label() before fusion
        so "Normal", "Calm", "CALM" etc. are all treated identically.
        The final behavior is ALWAYS either "aggressive" or "calm" — never "unknown".
        """
        # --- extract and validate pose branch ---
        pose_behavior = "calm"
        pose_confidence = 0.0
        pose_valid = False
        if pose_result is not None:
            raw = normalize_behavior_label(pose_result.behavior)  # always "aggressive"/"calm"
            if pose_result.confidence > 0:
                pose_behavior = raw
                pose_confidence = pose_result.confidence
                pose_valid = True
            else:
                pose_behavior = raw  # confidence=0 means no keypoints found

        # --- extract and validate sound branch ---
        sound_behavior = "calm"
        sound_confidence = 0.0
        sound_valid = False
        if sound_result is not None:
            # check the `valid` flag so failed sound results are skipped
            if getattr(sound_result, 'valid', True) and sound_result.behavior is not None:
                raw = normalize_behavior_label(sound_result.behavior)
                if sound_result.confidence > 0:
                    sound_behavior = raw
                    sound_confidence = sound_result.confidence
                    sound_valid = True
                else:
                    sound_behavior = raw
            else:
                err = getattr(sound_result, 'error', 'unknown error')
                logger.warning(f"Sound classifier result is invalid: {err}")

        # --- fusion logic ---
        if pose_valid and sound_valid:
            # Both branches produced meaningful results
            if pose_behavior == sound_behavior:
                # Agreement — boost confidence slightly
                final_behavior = pose_behavior
                final_confidence = min(1.0, (pose_confidence * 0.5 + sound_confidence * 0.5) * 1.1)
            else:
                # Disagreement — higher-confidence branch wins
                if pose_confidence >= sound_confidence:
                    final_behavior = pose_behavior
                    final_confidence = pose_confidence * 0.7
                else:
                    final_behavior = sound_behavior
                    final_confidence = sound_confidence * 0.7
            behavior_source = "fused"

        elif pose_valid:
            # sound failed; use pose-only result
            final_behavior = pose_behavior
            final_confidence = pose_confidence * 0.8
            behavior_source = "pose_only"

        elif sound_valid:
            # pose failed; use sound-only result
            final_behavior = sound_behavior
            final_confidence = sound_confidence * 0.8
            behavior_source = "sound_only"

        else:
            # Both failed — default to "calm" with zero confidence.
            # "unknown" is NEVER used as a behavior label.
            final_behavior = "calm"
            final_confidence = 0.0
            behavior_source = "none"

        return BehaviorResult(
            behavior=final_behavior,
            confidence=final_confidence,
            pose_behavior=pose_behavior,
            pose_confidence=pose_confidence,
            sound_behavior=sound_behavior,
            sound_confidence=sound_confidence,
            behavior_source=behavior_source,
        )
    
    def _smooth_behavior(self, current: BehaviorResult) -> BehaviorResult:
        """Apply temporal smoothing to behavior classification."""
        self.behavior_buffer.append(current)

        # Keep only recent results
        if len(self.behavior_buffer) > BEHAVIOR_SMOOTHING_WINDOW:
            self.behavior_buffer.pop(0)

        # Only count frames with a real prediction (confidence > 0)
        valid_behaviors = [b for b in self.behavior_buffer if b.behavior in ("aggressive", "calm")]

        if not valid_behaviors:
            # Nothing confident in the window — default to calm (never "unknown")
            return BehaviorResult(
                behavior="calm",
                confidence=0.0,
                pose_behavior=current.pose_behavior,
                pose_confidence=current.pose_confidence,
                sound_behavior=current.sound_behavior,
                sound_confidence=current.sound_confidence,
                behavior_source=current.behavior_source,
            )

        aggressive_count = sum(1 for b in valid_behaviors if b.behavior == "aggressive")
        calm_count = len(valid_behaviors) - aggressive_count

        if aggressive_count > calm_count:
            smoothed_behavior = "aggressive"
            smoothed_confidence = float(np.mean(
                [b.confidence for b in valid_behaviors if b.behavior == "aggressive"]
            ))
        else:
            smoothed_behavior = "calm"
            smoothed_confidence = float(np.mean(
                [b.confidence for b in valid_behaviors if b.behavior == "calm"]
            )) if calm_count > 0 else 0.0

        return BehaviorResult(
            behavior=smoothed_behavior,
            confidence=smoothed_confidence,
            pose_behavior=current.pose_behavior,
            pose_confidence=current.pose_confidence,
            sound_behavior=current.sound_behavior,
            sound_confidence=current.sound_confidence,
            behavior_source=current.behavior_source,
        )
    
    def _annotate_frame(self, frame: np.ndarray, detection: DetectionResult,
                       behavior: BehaviorResult) -> np.ndarray:
        """Create annotated frame with all detection and classification info"""
        annotated = frame.copy()
        
        # Draw detection boxes
        if detection.elephant_detected:
            for det in detection.detections:
                x1, y1, x2, y2 = det.bbox
                
                # Color based on behavior (lowercase comparison)
                if behavior.behavior == "aggressive":
                    box_color = (0, 0, 255)  # Red
                else:
                    box_color = (0, 255, 0)  # Green
                
                # Different shade for calf
                if det.class_name == "calf":
                    box_color = tuple(int(c * 0.7) for c in box_color)
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                
                # Label
                label = f"{det.class_name}: {det.confidence:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Info panel
        panel_height = 120
        cv2.rectangle(annotated, (0, 0), (350, panel_height), (0, 0, 0), -1)
        
        # Detection status
        status = "ELEPHANT DETECTED" if detection.elephant_detected else "NO ELEPHANT"
        status_color = (0, 255, 0) if detection.elephant_detected else (128, 128, 128)
        cv2.putText(annotated, status, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        if detection.elephant_detected:
            # Count info
            count_text = f"Adults: {detection.adult_count} | Calves: {detection.calf_count}"
            cv2.putText(annotated, count_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Group classification
            group_text = f"Group: {detection.group_classification.upper()}"
            cv2.putText(annotated, group_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Behavior (lowercase comparison; .upper() for display text)
            behavior_color = (0, 0, 255) if behavior.behavior == "aggressive" else (0, 255, 0)
            behavior_text = f"Behavior: {behavior.behavior.upper()} ({behavior.confidence:.2f})"
            cv2.putText(annotated, behavior_text, (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, behavior_color, 2)
            
            # Sub-classifications
            sub_text = f"Pose: {behavior.pose_behavior} | Sound: {behavior.sound_behavior}"
            cv2.putText(annotated, sub_text, (10, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return annotated
    
    def _create_video_result(self, video_path: str, duration: float,
                            total_frames: int, processed_frames: int,
                            fps: float, resolution: Tuple[int, int],
                            frame_results: List[FrameResult],
                            processing_time: float) -> VideoResult:
        """Create aggregate video result"""
        
        # Calculate dominant values
        elephant_detected = any(f.elephant_detected for f in frame_results)
        
        # Most common group classification
        groups = [f.group_classification for f in frame_results if f.elephant_detected]
        dominant_group = max(set(groups), key=groups.count) if groups else "none"
        
        # Most common behavior among elephant-detected frames
        behaviors = [f.behavior for f in frame_results if f.elephant_detected]
        dominant_behavior = max(set(behaviors), key=behaviors.count) if behaviors else "calm"

        # Compute dominant source and average confidence
        sources = [f.behavior_source for f in frame_results if f.elephant_detected]
        dominant_source = max(set(sources), key=sources.count) if sources else "none"

        conf_values = [
            f.behavior_confidence for f in frame_results
            if f.elephant_detected and f.behavior_confidence > 0
        ]
        avg_confidence = float(sum(conf_values) / len(conf_values)) if conf_values else 0.0
        
        # Max counts
        max_adult = max((f.adult_count for f in frame_results), default=0)
        max_calf = max((f.calf_count for f in frame_results), default=0)

        return VideoResult(
            video_path=video_path,
            duration=duration,
            total_frames=total_frames,
            processed_frames=processed_frames,
            fps=fps,
            resolution=resolution,
            elephant_detected=elephant_detected,
            dominant_group_classification=dominant_group,
            dominant_behavior=dominant_behavior,
            max_adult_count=max_adult,
            max_calf_count=max_calf,
            frame_results=frame_results,
            processing_time=processing_time,
            dominant_behavior_source=dominant_source,         # FIX
            dominant_behavior_confidence=avg_confidence,      # FIX
        )
    
    def _save_results(self, video_path: str, result: VideoResult):
        """Save results to JSON file"""
        output_file = OUTPUT_DIR / f"results_{Path(video_path).stem}.json"
        
        # Convert to serializable format
        result_dict = {
            'video_path': result.video_path,
            'duration': result.duration,
            'total_frames': result.total_frames,
            'processed_frames': result.processed_frames,
            'fps': result.fps,
            'resolution': result.resolution,
            'elephant_detected': result.elephant_detected,
            'dominant_group_classification': result.dominant_group_classification,
            'dominant_behavior': result.dominant_behavior,
            'max_adult_count': result.max_adult_count,
            'max_calf_count': result.max_calf_count,
            'processing_time': result.processing_time,
            'frame_results': [asdict(f) for f in result.frame_results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def _print_summary(self, result: VideoResult):
        """Print clean, standardized final analysis summary."""
        print("\n" + "=" * 48)
        print("FINAL ANALYSIS RESULTS")
        print("=" * 48)
        print(f"Elephant Detected: {'YES' if result.elephant_detected else 'NO'}")
        if result.elephant_detected:
            print(f"Group Type:        {result.dominant_group_classification.upper()}")
            print(f"Max Adults:        {result.max_adult_count}")
            print(f"Max Calves:        {result.max_calf_count}")
            # dominant_behavior is always "aggressive" or "calm" — .upper() gives AGGRESSIVE/CALM
            print(f"Behavior:          {result.dominant_behavior.upper()}")
            print(f"Behavior Source:   {result.dominant_behavior_source.upper()}")
            print(f"Confidence:        {result.dominant_behavior_confidence:.2f}")
        print(f"Processing Time:   {result.processing_time:.2f} seconds")
        print("=" * 48 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Elephant Detection and Behavior Classification Pipeline'
    )
    parser.add_argument('video', type=str, help='Path to input video')
    parser.add_argument('--output', '-o', type=str, help='Path for output video')
    parser.add_argument('--preview', '-p', action='store_true', help='Show real-time preview')
    parser.add_argument('--no-sound', action='store_true', help='Disable sound classification')
    parser.add_argument('--no-pose', action='store_true', help='Disable pose classification')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IntegratedPipeline(
        enable_sound=not args.no_sound,
        enable_pose=not args.no_pose
    )
    
    # Process video
    result = pipeline.process_video(
        args.video,
        output_path=args.output,
        show_preview=args.preview
    )
    
    return result


if __name__ == "__main__":
    main()
