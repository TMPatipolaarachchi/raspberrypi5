# Quick Reference: Code Changes by File

## 1. config.py

### Added Lines (after line 73):
```python
# Behavior label normalization mapping
# Maps various label formats to standard format
BEHAVIOR_LABEL_MAPPING = {
    # Aggressive variations
    "aggressive": "aggressive",
    "agitated": "aggressive",
    "threat": "aggressive",
    "threatening": "aggressive",
    "trumpet": "aggressive",
    "trumpeting": "aggressive",
    "charging": "aggressive",
    "attack": "aggressive",
    # Calm variations
    "calm": "calm",
    "normal": "calm",
    "relaxed": "calm",
    "passive": "calm",
    "feeding": "calm",
    "resting": "calm",
    "neutral": "calm"
}
```

---

## 2. sound_classifier.py

### Import Changes (line ~23):
```python
from config import (
    SOUND_CNN_LSTM_MODEL,
    SOUND_RF_MODEL,
    SOUND_XGB_MODEL,
    SOUND_SCALER,
    SOUND_HYBRID_SCALER,
    SOUND_LABEL_ENCODER,
    SOUND_FEATURE_ORDER,
    SOUND_HYBRID_FEATURE_ORDER,
    AUDIO_SAMPLE_RATE,
    AUDIO_DURATION,
    AUDIO_HOP_LENGTH,
    AUDIO_N_FFT,
    AUDIO_N_MFCC,
    BEHAVIOR_LABEL_MAPPING  # <-- ADDED
)
```

### Method: _predict_cnn_lstm (line ~238):

**BEFORE:**
```python
def _predict_cnn_lstm(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
    try:
        # Prepare mel spectrogram for CNN-LSTM
        mel_spec = librosa.feature.melspectrogram(...)
        
        # ... processing ...
        
        prediction = self.cnn_lstm_model.predict(mel_input, verbose=0)
        
        if prediction.shape[-1] == 1:
            prob = float(prediction[0][0])  # <-- COULD FAIL HERE
            pred_class = 1 if prob > 0.5 else 0
            confidence = prob if pred_class == 1 else 1 - prob
        else:
            pred_class = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][pred_class])
        
        behavior = self.label_encoder.inverse_transform([pred_class])[0]
        behavior_mapped = "aggressive" if behavior.lower() in ["aggressive", "agitated", "threat", "trumpet"] else "calm"
        
        return behavior_mapped, confidence
        
    except Exception as e:
        logger.error(f"Error in CNN-LSTM prediction: {e}")
        return "calm", 0.0
```

**AFTER:**
```python
def _predict_cnn_lstm(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
    try:
        # Check if audio is valid
        if audio is None or len(audio) == 0:
            logger.warning("Empty audio input for CNN-LSTM prediction")
            return "calm", 0.0
        
        # Prepare mel spectrogram for CNN-LSTM
        mel_spec = librosa.feature.melspectrogram(...)
        
        # ... processing ...
        
        prediction = self.cnn_lstm_model.predict(mel_input, verbose=0)
        
        # Defensive check for prediction shape
        if prediction is None or prediction.size == 0:
            logger.warning("Empty prediction from CNN-LSTM model")
            return "calm", 0.0
        
        # Flatten prediction if needed for consistent access
        prediction = np.atleast_2d(prediction)
        
        # Determine if binary or multi-class
        if prediction.shape[-1] == 1:
            # Binary classification with sigmoid
            prob = float(prediction[0, 0])
            pred_class = 1 if prob > 0.5 else 0
            confidence = prob if pred_class == 1 else 1 - prob
        elif len(prediction.shape) == 2 and prediction.shape[0] == 1:
            # Multi-class with softmax or single output
            if prediction.shape[1] == 1:
                prob = float(prediction[0, 0])
                pred_class = 1 if prob > 0.5 else 0
                confidence = prob if pred_class == 1 else 1 - prob
            else:
                pred_class = int(np.argmax(prediction[0]))
                confidence = float(prediction[0, pred_class])
        else:
            # Unexpected shape - try to handle gracefully
            logger.warning(f"Unexpected prediction shape: {prediction.shape}")
            prediction_flat = prediction.flatten()
            if len(prediction_flat) == 1:
                prob = float(prediction_flat[0])
                pred_class = 1 if prob > 0.5 else 0
                confidence = prob if pred_class == 1 else 1 - prob
            else:
                pred_class = int(np.argmax(prediction_flat))
                confidence = float(prediction_flat[pred_class])
        
        # Validate pred_class is within label encoder range
        if pred_class >= len(self.label_encoder.classes_):
            logger.warning(f"Predicted class {pred_class} out of range, defaulting to 0")
            pred_class = 0
        
        # Get behavior label from encoder and normalize
        behavior = self.label_encoder.inverse_transform([pred_class])[0]
        behavior_mapped = self._normalize_behavior_label(behavior)
        
        return behavior_mapped, confidence
        
    except Exception as e:
        logger.error(f"Error in CNN-LSTM prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "calm", 0.0
```

### Method: _predict_ml_ensemble (line ~291):

**BEFORE:**
```python
def _predict_ml_ensemble(self, features: Dict) -> Tuple[str, float]:
    try:
        # Prepare feature vector
        feature_vector = []
        for feature_name in self.feature_order:
            feature_vector.append(features.get(feature_name, 0.0))
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_array)
        
        # RF and XGB predictions...
        
        behavior = self.label_encoder.inverse_transform([final_pred])[0]
        behavior_mapped = "aggressive" if behavior.lower() in ["aggressive", "agitated", "threat", "trumpet"] else "calm"
        
        return behavior_mapped, float(final_conf)
        
    except Exception as e:
        logger.error(f"Error in ML ensemble prediction: {e}")
        return "calm", 0.0
```

**AFTER:**
```python
def _predict_ml_ensemble(self, features: Dict) -> Tuple[str, float]:
    try:
        # Check if features are valid
        if not features or len(features) == 0:
            logger.warning("Empty features for ML ensemble prediction")
            return "calm", 0.0
        
        # Prepare feature vector
        feature_vector = []
        for feature_name in self.feature_order:
            feature_vector.append(features.get(feature_name, 0.0))
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Check for NaN or inf values
        if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
            logger.warning("NaN or inf values in features, replacing with 0")
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        feature_scaled = self.scaler.transform(feature_array)
        
        # RF and XGB predictions...
        
        # Convert prediction to int if needed
        final_pred = int(final_pred)
        
        # Validate pred_class is within label encoder range
        if final_pred >= len(self.label_encoder.classes_):
            logger.warning(f"Predicted class {final_pred} out of range, defaulting to 0")
            final_pred = 0
        elif final_pred < 0:
            logger.warning(f"Predicted class {final_pred} is negative, defaulting to 0")
            final_pred = 0
        
        # Get behavior label and normalize
        behavior = self.label_encoder.inverse_transform([final_pred])[0]
        behavior_mapped = self._normalize_behavior_label(behavior)
        
        return behavior_mapped, float(final_conf)
        
    except Exception as e:
        logger.error(f"Error in ML ensemble prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "calm", 0.0
```

### NEW Method: _normalize_behavior_label (line ~400):

```python
def _normalize_behavior_label(self, label: str) -> str:
    """
    Normalize behavior labels to standard format: 'aggressive' or 'calm'
    Uses centralized mapping from config.py
    """
    if not label or not isinstance(label, str):
        return "calm"
    
    label_lower = label.lower().strip()
    
    # Use centralized mapping from config
    normalized = BEHAVIOR_LABEL_MAPPING.get(label_lower)
    
    if normalized:
        return normalized
    else:
        # Default to calm for unknown labels
        logger.warning(f"Unknown behavior label '{label}', defaulting to 'calm'")
        return "calm"
```

### Updated Method: _combine_predictions (line ~420):

**BEFORE:**
```python
def _combine_predictions(self, cnn_lstm_result: Tuple[str, float], 
                        ml_result: Tuple[str, float]) -> Tuple[str, float]:
    cnn_behavior, cnn_conf = cnn_lstm_result
    ml_behavior, ml_conf = ml_result
    
    # Weight: CNN-LSTM 0.6, ML 0.4
    cnn_weight = 0.6
    ml_weight = 0.4
    
    # ... combination logic ...
```

**AFTER:**
```python
def _combine_predictions(self, cnn_lstm_result: Tuple[str, float], 
                        ml_result: Tuple[str, float]) -> Tuple[str, float]:
    cnn_behavior, cnn_conf = cnn_lstm_result
    ml_behavior, ml_conf = ml_result
    
    # Normalize behaviors just in case
    cnn_behavior = self._normalize_behavior_label(cnn_behavior)
    ml_behavior = self._normalize_behavior_label(ml_behavior)
    
    # Weight: CNN-LSTM 0.6, ML 0.4
    cnn_weight = 0.6
    ml_weight = 0.4
    
    # ... combination logic ...
```

---

## 3. pose_classifier.py

### Import Changes (line ~15):
```python
from config import (
    POSE_YOLO_MODEL,
    POSE_CLASSIFIER,
    POSE_SCALER,
    POSE_LABEL_ENCODER,
    POSE_FEATURE_ORDER,
    POSE_CONFIDENCE,
    POSE_IOU_THRESHOLD,
    BEHAVIOR_CLASSES,
    BEHAVIOR_LABEL_MAPPING  # <-- ADDED
)
```

### Method: classify (line ~138):

**BEFORE:**
```python
# Classify behavior
prediction = self.classifier.predict(feature_scaled)[0]
probabilities = self.classifier.predict_proba(feature_scaled)[0]
confidence = np.max(probabilities)

# Decode label
behavior = self.label_encoder.inverse_transform([prediction])[0]

# Map to standard output
behavior_mapped = "aggressive" if behavior.lower() in ["aggressive", "agitated", "threat"] else "calm"

return PoseResult(
    behavior=behavior_mapped,
    confidence=float(confidence),
    pose_keypoints=keypoints,
    features=features
)
```

**AFTER:**
```python
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

# Normalize to standard output using centralized mapping
behavior_mapped = self._normalize_behavior_label(behavior)

return PoseResult(
    behavior=behavior_mapped,
    confidence=float(confidence),
    pose_keypoints=keypoints,
    features=features
)
```

### NEW Method: _normalize_behavior_label (line ~167):

```python
def _normalize_behavior_label(self, label: str) -> str:
    """
    Normalize behavior labels to standard format: 'aggressive' or 'calm'
    Uses centralized mapping from config.py
    """
    if not label or not isinstance(label, str):
        return "calm"
    
    label_lower = label.lower().strip()
    
    # Use centralized mapping from config
    normalized = BEHAVIOR_LABEL_MAPPING.get(label_lower)
    
    if normalized:
        return normalized
    else:
        # Default to calm for unknown labels
        logger.warning(f"Unknown behavior label '{label}', defaulting to 'calm'")
        return "calm"
```

---

## Summary of Changes

### Files Modified: 3
1. ✅ config.py - Added centralized label mapping
2. ✅ sound_classifier.py - Fixed both errors, added defensive checks
3. ✅ pose_classifier.py - Added defensive checks, consistent labeling

### Lines Added: ~200
### Lines Modified: ~50

### Key Improvements:
- ✅ CNN-LSTM index error fixed with shape validation
- ✅ Label mismatch error fixed with normalization
- ✅ All labels standardized to "aggressive" or "calm"
- ✅ Defensive checks for empty/invalid data
- ✅ Better error logging with tracebacks
- ✅ Centralized configuration for maintainability
