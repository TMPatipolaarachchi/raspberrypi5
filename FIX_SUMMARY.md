# Sound Classifier Bug Fixes - Summary

## Files Modified

### 1. **sound_classifier.py**
   - Fixed CNN-LSTM prediction index error
   - Fixed ML ensemble label mismatch error
   - Added comprehensive defensive checks
   - Added centralized label normalization

### 2. **pose_classifier.py**
   - Added defensive checks for predictions
   - Updated to use centralized label normalization
   - Fixed potential label mismatch issues

### 3. **config.py**
   - Added `BEHAVIOR_LABEL_MAPPING` dictionary for centralized label normalization
   - Maps all variations (Normal, Calm, Aggressive, etc.) to standard format

---

## Issues Identified and Fixed

### Issue 1: CNN-LSTM "list index out of range"

**Location**: `sound_classifier.py` line ~267

**Root Cause**:
- The code accessed `prediction[0][0]` assuming a specific shape
- If prediction had shape `(1,)` instead of `(1, 1)`, it would fail
- No defensive handling for empty or malformed predictions

**Fix Applied**:
- Added empty audio check before processing
- Added defensive check for prediction shape
- Use `np.atleast_2d()` to ensure consistent 2D array
- Handle multiple prediction shapes: binary (1,), (1,1), multi-class (1, n)
- Added prediction class range validation
- Added detailed error logging with traceback

**Code Changes**:
```python
# Check if audio is valid
if audio is None or len(audio) == 0:
    logger.warning("Empty audio input for CNN-LSTM prediction")
    return "calm", 0.0

# Defensive check for prediction shape
if prediction is None or prediction.size == 0:
    logger.warning("Empty prediction from CNN-LSTM model")
    return "calm", 0.0

# Flatten prediction if needed for consistent access
prediction = np.atleast_2d(prediction)

# Handle different prediction shapes
if prediction.shape[-1] == 1:
    # Binary classification with sigmoid
    prob = float(prediction[0, 0])
    pred_class = 1 if prob > 0.5 else 0
    confidence = prob if pred_class == 1 else 1 - prob
elif len(prediction.shape) == 2 and prediction.shape[0] == 1:
    # Multi-class or single output
    if prediction.shape[1] == 1:
        prob = float(prediction[0, 0])
        pred_class = 1 if prob > 0.5 else 0
        confidence = prob if pred_class == 1 else 1 - prob
    else:
        # Multi-class softmax
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
```

---

### Issue 2: ML Ensemble "previously unseen labels: ['Normal']"

**Location**: `sound_classifier.py` line ~327

**Root Cause**:
- The label encoder was trained with labels like "Normal", "Aggressive"
- When `inverse_transform()` returned "Normal", it was passed somewhere expecting "calm"
- The RF/XGB models might predict class indices not in label encoder's range
- The mapping logic only covered some label variations
- No validation that predicted class index is valid

**Fix Applied**:
- Added empty features check
- Added NaN/inf value checking and replacement
- Convert prediction to int explicitly
- Validate prediction is within label encoder range
- Use centralized label normalization function
- Added detailed error logging with traceback

**Code Changes**:
```python
# Check if features are valid
if not features or len(features) == 0:
    logger.warning("Empty features for ML ensemble prediction")
    return "calm", 0.0

# Check for NaN or inf values
if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
    logger.warning("NaN or inf values in features, replacing with 0")
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

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
```

---

### Issue 3: Inconsistent Label Formats

**Root Cause**:
- Training data used inconsistent label names: "Normal", "Calm", "Aggressive", etc.
- Each model might have different label encodings
- Hardcoded label mappings scattered across multiple files
- No centralized label normalization

**Fix Applied**:
- Created centralized `BEHAVIOR_LABEL_MAPPING` in `config.py`
- Added `_normalize_behavior_label()` method to both classifiers
- Maps all known variations to standard format: "aggressive" or "calm"
- Default to "calm" for unknown labels with warning

**Centralized Label Mapping** (in `config.py`):
```python
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

**Normalization Function** (added to both classifiers):
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

## Why the Errors Happened

### CNN-LSTM Index Error:
1. Model output shape varies based on training configuration
2. Binary classification can output shape `(1,)`, `(1,1)`, or `(batch, 1)`
3. Code assumed specific shape without validation
4. No error handling for edge cases (empty audio, invalid predictions)

### Label Mismatch Error:
1. Models were trained with different label encodings
2. Label encoder had "Normal" but downstream code expected "calm"
3. No validation that predicted class indices exist in encoder
4. RF/XGB might predict out-of-range class indices
5. No centralized label normalization strategy

---

## Testing Recommendations

After applying these fixes, test with:

1. **Normal video with sound**: Should work without errors
2. **Video without audio**: Should gracefully default to calm
3. **Very short audio clips**: Should handle edge cases
4. **Silent audio**: Should not crash with empty features

---

## Additional Defensive Checks Added

✅ Empty audio input validation  
✅ Empty prediction validation  
✅ Prediction shape normalization  
✅ Class index range validation  
✅ NaN/inf feature value handling  
✅ Empty features dictionary validation  
✅ Detailed error logging with tracebacks  
✅ Graceful fallback to "calm" on any error  

---

## Final Behavior Label Standard

All behavior classifications now output only two values:
- **"aggressive"** - for threatening, attacking, or agitated behavior
- **"calm"** - for normal, relaxed, feeding, or passive behavior

Labels are normalized at the point where they're decoded from the label encoder, ensuring consistency throughout the pipeline.
