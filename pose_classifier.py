"""
Pose-Based Behavior Classification Module
Analyzes elephant poses from video frames to classify behavior as aggressive or calm
"""

import cv2
import numpy as np
import json
import joblib
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import math

from config import (
    POSE_YOLO_MODEL,
    POSE_CLASSIFIER,
    POSE_SCALER,
    POSE_LABEL_ENCODER,
    POSE_FEATURE_ORDER,
    POSE_CONFIDENCE,
    POSE_IOU_THRESHOLD,
    BEHAVIOR_CLASSES,
    BEHAVIOR_LABEL_MAPPING,
    normalize_behavior_label,   # FIX: use shared normalizer so labels are proper case
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    """Data class for pose-based behavior classification result"""
    behavior: str  # "aggressive" or "calm"
    confidence: float
    pose_keypoints: Optional[np.ndarray] = None
    features: Optional[Dict] = None


class PoseBehaviorClassifier:
    """
    Pose-based behavior classifier using YOLOv8-pose for keypoint detection
    and ML model for behavior classification
    """
    
    # Keypoint indices for elephant pose (based on COCO-like format adapted for elephants)
    # These indices represent different body parts
    KEYPOINT_MAPPING = {
        'head': 0,
        'left_ear': 1,
        'right_ear': 2,
        'trunk_base': 3,
        'trunk_tip': 4,
        'front_left_leg_top': 5,
        'front_left_leg_bottom': 6,
        'front_right_leg_top': 7,
        'front_right_leg_bottom': 8,
        'back_left_leg_top': 9,
        'back_left_leg_bottom': 10,
        'back_right_leg_top': 11,
        'back_right_leg_bottom': 12,
        'tail_base': 13,
        'tail_tip': 14,
        'body_center': 15,
        'hip': 16
    }
    
    def __init__(self, pose_model_path: str = None, classifier_path: str = None):
        """
        Initialize the pose-based behavior classifier
        
        Args:
            pose_model_path: Path to YOLOv8-pose model
            classifier_path: Path to ML classifier model
        """
        self.pose_model_path = pose_model_path or str(POSE_YOLO_MODEL)
        self.classifier_path = classifier_path or str(POSE_CLASSIFIER)
        
        # Load pose detection model
        logger.info(f"Loading pose detection model from {self.pose_model_path}")
        self.pose_model = YOLO(self.pose_model_path)
        self.pose_model.overrides['verbose'] = False
        
        # Load ML classifier and preprocessing components
        logger.info(f"Loading pose classifier from {self.classifier_path}")
        self.classifier = joblib.load(self.classifier_path)
        self.scaler = joblib.load(str(POSE_SCALER))
        self.label_encoder = joblib.load(str(POSE_LABEL_ENCODER))
        
        # Load feature order
        with open(str(POSE_FEATURE_ORDER), 'r') as f:
            self.feature_config = json.load(f)
            self.feature_order = self.feature_config['feature_order']
        
        logger.info("Pose behavior classifier initialized successfully")
    
    def classify(self, frame: np.ndarray) -> PoseResult:
        """
        Classify elephant behavior based on pose analysis
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            PoseResult object with behavior classification
        """
        # Detect pose keypoints
        keypoints = self._detect_pose(frame)
        
        if keypoints is None or len(keypoints) == 0:
            # No pose detected — return structured default (confidence=0 signals
            # "not enough data" to the fusion layer without crashing the pipeline)
            return PoseResult(
                behavior="calm",   # lowercase: only "aggressive" or "calm" allowed
                confidence=0.0,
                pose_keypoints=None,
                features=None
            )

        # Compute features from keypoints
        features = self._compute_features(keypoints)

        if features is None:
            return PoseResult(
                behavior="calm",   # lowercase: only "aggressive" or "calm" allowed
                confidence=0.0,
                pose_keypoints=keypoints,
                features=None
            )
        
        # Prepare features for classification
        feature_vector = self._prepare_feature_vector(features)
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Classify behavior
        prediction = self.classifier.predict(feature_scaled)[0]
        probabilities = self.classifier.predict_proba(feature_scaled)[0]
        confidence = np.max(probabilities)
        
        # Convert prediction to int if needed
        prediction = int(prediction)
        
        # Validate prediction is within label encoder range
        if prediction >= len(self.label_encoder.classes_):
            logger.warning(f"Predicted class {prediction} out of range, defaulting to 0")
            prediction = 0
        elif prediction < 0:
            logger.warning(f"Predicted class {prediction} is negative, defaulting to 0")
            prediction = 0
        
        # Decode label
        behavior = self.label_encoder.inverse_transform([prediction])[0]

        # FIX: normalize to proper-case standard label using shared function
        behavior_mapped = normalize_behavior_label(behavior)  # "Normal" -> "Calm", etc.
        
        return PoseResult(
            behavior=behavior_mapped,
            confidence=float(confidence),
            pose_keypoints=keypoints,
            features=features
        )
    
    def _normalize_behavior_label(self, label: str) -> str:
        """
        Instance-level wrapper around the module-level normalize_behavior_label().
        Kept for backward compatibility; delegates all logic to config.py.
        Returns proper-case labels: "Aggressive", "Calm", or the raw stripped label
        when the value is not recognised.
        """
        return normalize_behavior_label(label)
    
    def _detect_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose keypoints from frame using YOLOv8-pose
        """
        results = self.pose_model(
            frame,
            conf=POSE_CONFIDENCE,
            iou=POSE_IOU_THRESHOLD,
            verbose=False
        )
        
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                # Get the first detected pose (highest confidence elephant)
                keypoints = result.keypoints.data[0].cpu().numpy()
                return keypoints
        
        return None
    
    def _compute_features(self, keypoints: np.ndarray) -> Optional[Dict]:
        """
        Compute pose features from keypoints
        
        Features:
        - ear_spread: Distance between ears
        - trunk_length: Length of trunk
        - tail_length: Length of tail
        - leg_stance: Front and back leg positions
        - Various ratios and angles
        """
        try:
            # Extract keypoints (x, y, confidence)
            n_keypoints = min(len(keypoints), 17)
            
            # Create a mapping for available keypoints
            kp = {}
            for i in range(n_keypoints):
                kp[i] = keypoints[i][:2] if keypoints[i][2] > 0.1 else None
            
            features = {}
            
            # Calculate distances and features
            # Ear spread
            if kp.get(1) is not None and kp.get(2) is not None:
                features['ear_spread'] = self._distance(kp[1], kp[2])
            else:
                features['ear_spread'] = 0

            # Trunk length
            if kp.get(3) is not None and kp.get(4) is not None:
                features['trunk_length'] = self._distance(kp[3], kp[4])
            else:
                features['trunk_length'] = 0

            # Tail length
            if kp.get(13) is not None and kp.get(14) is not None:
                features['tail_length'] = self._distance(kp[13], kp[14])
            else:
                features['tail_length'] = 0

            # Front leg stance
            if kp.get(6) is not None and kp.get(8) is not None:
                features['front_leg_stance'] = self._distance(kp[6], kp[8])
            else:
                features['front_leg_stance'] = 0

            # Back leg stance
            if kp.get(10) is not None and kp.get(12) is not None:
                features['back_leg_stance'] = self._distance(kp[10], kp[12])
            else:
                features['back_leg_stance'] = 0

            # Calculate body reference for ratios
            body_ref = 1.0
            if kp.get(0) is not None and kp.get(16) is not None:
                body_ref = max(self._distance(kp[0], kp[16]), 1.0)

            # Ratios
            features['ear_ratio'] = features['ear_spread'] / body_ref
            features['trunk_ratio'] = features['trunk_length'] / body_ref
            features['front_leg_ratio'] = features['front_leg_stance'] / body_ref
            features['back_leg_ratio'] = features['back_leg_stance'] / body_ref
            features['tail_ratio'] = features['tail_length'] / body_ref
            features['body_ratio'] = body_ref / 100.0  # Normalize

            # Angles (normalized to [-1, 1])
            features['head_angle_norm'] = self._compute_angle_normalized(kp.get(0), kp.get(3), kp.get(15))
            features['trunk_angle_norm'] = self._compute_angle_normalized(kp.get(3), kp.get(4), (kp.get(3)[0] + 50, kp.get(3)[1]) if kp.get(3) is not None else None)
            features['front_leg_angle_norm'] = self._compute_angle_normalized(kp.get(5), kp.get(6), kp.get(7))
            features['back_leg_angle_norm'] = self._compute_angle_normalized(kp.get(9), kp.get(10), kp.get(11))
            features['ear_angle_norm'] = self._compute_angle_normalized(kp.get(1), kp.get(0), kp.get(2))
            features['tail_angle_norm'] = self._compute_angle_normalized(kp.get(13), kp.get(14), (kp.get(13)[0], kp.get(13)[1] + 50) if kp.get(13) is not None else None)

            return features
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return None
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _compute_angle_normalized(self, p1, p2, p3) -> float:
        """
        Compute the angle at p2 formed by p1-p2-p3, normalized to [-1, 1]
        """
        if p1 is None or p2 is None or p3 is None:
            return 0.0
        
        try:
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return float(np.clip(cos_angle, -1, 1))
        except:
            return 0.0
    
    def _prepare_feature_vector(self, features: Dict) -> np.ndarray:
        """
        Prepare feature vector in the correct order for the classifier
        """
        feature_vector = []
        for feature_name in self.feature_order:
            feature_vector.append(features.get(feature_name, 0.0))
        return np.array(feature_vector)
    
    def annotate_frame(self, frame: np.ndarray, keypoints: np.ndarray, 
                       behavior: str, confidence: float) -> np.ndarray:
        """
        Annotate frame with pose keypoints and behavior classification
        """
        annotated = frame.copy()
        
        # Draw keypoints
        if keypoints is not None:
            for i, kp in enumerate(keypoints):
                if len(kp) >= 3 and kp[2] > 0.3:  # Confidence threshold
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(annotated, (x, y), 5, (0, 255, 255), -1)
            
            # Draw skeleton connections
            skeleton = [
                (0, 1), (0, 2), (0, 3), (3, 4),  # Head, ears, trunk
                (0, 15), (15, 5), (15, 7), (15, 9), (15, 11),  # Body to legs
                (5, 6), (7, 8), (9, 10), (11, 12),  # Legs
                (15, 13), (13, 14)  # Tail
            ]
            
            for conn in skeleton:
                if conn[0] < len(keypoints) and conn[1] < len(keypoints):
                    if keypoints[conn[0]][2] > 0.3 and keypoints[conn[1]][2] > 0.3:
                        pt1 = tuple(map(int, keypoints[conn[0]][:2]))
                        pt2 = tuple(map(int, keypoints[conn[1]][:2]))
                        cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
        
        # Add behavior label
        color = (0, 0, 255) if behavior == "aggressive" else (0, 255, 0)
        label = f"Pose: {behavior.upper()} ({confidence:.2f})"
        cv2.putText(annotated, label, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated


if __name__ == "__main__":
    import sys
    
    classifier = PoseBehaviorClassifier()
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        frame = cv2.imread(image_path)
        if frame is not None:
            result = classifier.classify(frame)
            print(f"Behavior: {result.behavior}")
            print(f"Confidence: {result.confidence:.2f}")
            
            if result.pose_keypoints is not None:
                annotated = classifier.annotate_frame(
                    frame, result.pose_keypoints, 
                    result.behavior, result.confidence
                )
                cv2.imshow("Pose Classification", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"Could not load image: {image_path}")
    else:
        print("Pose Behavior Classifier initialized successfully.")
        print("Usage: python pose_classifier.py <image_path>")
