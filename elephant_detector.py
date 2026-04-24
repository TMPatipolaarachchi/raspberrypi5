"""
Elephant Detection Module
Detects elephants, classifies as adult/calf, counts them, and determines group classification
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from config import (
    ELEPHANT_DETECT_MODEL,
    DETECTION_CONFIDENCE,
    DETECTION_IOU_THRESHOLD,
    ELEPHANT_CLASSES,
    GROUP_INDIVIDUAL_MAX,
    GROUP_FAMILY_MIN_ADULTS,
    GROUP_FAMILY_MAX_ADULTS,
    GROUP_FAMILY_MIN_CALVES,
    NUM_THREADS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ElephantDetection:
    """Data class for individual elephant detection"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str  # "adult" or "calf"


@dataclass
class DetectionResult:
    """Data class for complete detection results"""
    elephant_detected: bool
    total_count: int
    adult_count: int
    calf_count: int
    group_classification: str  # "individual", "family", or "herd"
    detections: List[ElephantDetection]
    annotated_frame: Optional[np.ndarray] = None


class ElephantDetector:
    """
    Elephant Detection class using YOLOv8 model
    Handles detection, classification (adult/calf), counting, and group classification
    """
    
    def __init__(self, model_path: str = None, confidence: float = None):
        """
        Initialize the elephant detector
        
        Args:
            model_path: Path to the YOLOv8 model file
            confidence: Detection confidence threshold
        """
        self.model_path = model_path or str(ELEPHANT_DETECT_MODEL)
        self.confidence = confidence or DETECTION_CONFIDENCE
        
        logger.info(f"Loading elephant detection model from {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Optimize for Raspberry Pi
        self.model.overrides['verbose'] = False
        
        logger.info("Elephant detector initialized successfully")
    
    def detect(self, frame: np.ndarray, annotate: bool = True) -> DetectionResult:
        """
        Detect elephants in a frame
        
        Args:
            frame: Input frame (BGR format)
            annotate: Whether to return annotated frame
            
        Returns:
            DetectionResult object with all detection information
        """
        # Run detection
        results = self.model(
            frame,
            conf=self.confidence,
            iou=DETECTION_IOU_THRESHOLD,
            verbose=False
        )
        
        detections = []
        adult_count = 0
        calf_count = 0
        
        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name (adult or calf)
                    class_name = ELEPHANT_CLASSES.get(class_id, "adult")
                    
                    # Count by type
                    if class_name == "adult":
                        adult_count += 1
                    else:
                        calf_count += 1
                    
                    detections.append(ElephantDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    ))
        
        total_count = adult_count + calf_count
        elephant_detected = total_count > 0
        
        # Determine group classification
        group_classification = self._classify_group(adult_count, calf_count)
        
        # Create annotated frame if requested
        annotated_frame = None
        if annotate and elephant_detected:
            annotated_frame = self._annotate_frame(frame, detections, group_classification)
        
        return DetectionResult(
            elephant_detected=elephant_detected,
            total_count=total_count,
            adult_count=adult_count,
            calf_count=calf_count,
            group_classification=group_classification,
            detections=detections,
            annotated_frame=annotated_frame
        )
    
    def _classify_group(self, adult_count: int, calf_count: int) -> str:
        """
        Classify the group based on elephant counts
        
        Rules:
        - Individual: 1 elephant (no matter adult or calf)
        - Family: 2 adults with at least 1 calf
        - Herd: Any other combination (more than family or just adults/calves)
        """
        total = adult_count + calf_count
        
        if total == 0:
            return "none"
        elif total <= GROUP_INDIVIDUAL_MAX:
            return "individual"
        elif (adult_count >= GROUP_FAMILY_MIN_ADULTS and 
              adult_count <= GROUP_FAMILY_MAX_ADULTS and 
              calf_count >= GROUP_FAMILY_MIN_CALVES):
            return "family"
        else:
            return "herd"
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[ElephantDetection], 
                        group_classification: str) -> np.ndarray:
        """
        Annotate the frame with detection boxes and information
        """
        annotated = frame.copy()
        
        # Colors for different classes
        colors = {
            "adult": (0, 255, 0),    # Green
            "calf": (255, 165, 0)     # Orange
        }
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors.get(detection.class_name, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add group classification info
        info_text = f"Group: {group_classification} | Adults: {sum(1 for d in detections if d.class_name == 'adult')} | Calves: {sum(1 for d in detections if d.class_name == 'calf')}"
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated


def process_frame(detector: ElephantDetector, frame: np.ndarray) -> DetectionResult:
    """
    Convenience function to process a single frame
    """
    return detector.detect(frame)


if __name__ == "__main__":
    # Test the detector
    import sys
    
    detector = ElephantDetector()
    
    # Test with a sample image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        frame = cv2.imread(image_path)
        if frame is not None:
            result = detector.detect(frame)
            print(f"Elephant Detected: {result.elephant_detected}")
            print(f"Total Count: {result.total_count}")
            print(f"Adults: {result.adult_count}, Calves: {result.calf_count}")
            print(f"Group Classification: {result.group_classification}")
            
            if result.annotated_frame is not None:
                cv2.imshow("Detection Result", result.annotated_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"Could not load image: {image_path}")
    else:
        print("Elephant Detector initialized successfully.")
        print("Usage: python elephant_detector.py <image_path>")
