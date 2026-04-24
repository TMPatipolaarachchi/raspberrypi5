"""
Diagnostic script to inspect sound-based models and identify label mismatches
"""
import numpy as np
import json
import sys
import traceback
from pathlib import Path

print("=" * 70)
print("SOUND MODEL DIAGNOSTICS")
print("=" * 70)

# Try to load models
BASE_DIR = Path(__file__).parent
SOUND_MODEL_DIR = BASE_DIR / "sound-based" / "model"

# 1. Check label encoder
print("\n1. LABEL ENCODER INSPECTION:")
print("-" * 70)
try:
    import joblib
    label_encoder_path = SOUND_MODEL_DIR / "label_encoder.pkl"
    label_encoder = joblib.load(label_encoder_path)
    print(f"✓ Label encoder loaded from: {label_encoder_path}")
    print(f"  Classes: {label_encoder.classes_}")
    print(f"  Number of classes: {len(label_encoder.classes_)}")
    
    # Test inverse transform
    for i in range(len(label_encoder.classes_)):
        label = label_encoder.inverse_transform([i])[0]
        print(f"  Index {i} -> '{label}'")
        
except Exception as e:
    print(f"✗ Error loading label encoder: {e}")
    traceback.print_exc()

# 2. Check CNN-LSTM model
print("\n2. CNN-LSTM MODEL INSPECTION:")
print("-" * 70)
try:
    import tensorflow as tf
    from tensorflow import keras
    
    model_path = SOUND_MODEL_DIR / "cnn_lstm_model.h5"
    model = keras.models.load_model(str(model_path))
    print(f"✓ CNN-LSTM model loaded from: {model_path}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Output units: {model.output_shape[-1]}")
    
    # Determine if binary or multi-class
    if model.output_shape[-1] == 1:
        print(f"  Type: BINARY classification (sigmoid output)")
    else:
        print(f"  Type: MULTI-CLASS classification (softmax output)")
        
    # Create dummy input and test prediction
    dummy_input = np.random.randn(1, *model.input_shape[1:])
    prediction = model.predict(dummy_input, verbose=0)
    print(f"  Test prediction shape: {prediction.shape}")
    print(f"  Test prediction values: {prediction}")
    
except Exception as e:
    print(f"✗ Error loading CNN-LSTM model: {e}")
    traceback.print_exc()

# 3. Check RF model
print("\n3. RANDOM FOREST MODEL INSPECTION:")
print("-" * 70)
try:
    import joblib
    rf_path = SOUND_MODEL_DIR / "rf_model.pkl"
    rf_model = joblib.load(rf_path)
    print(f"✓ RF model loaded from: {rf_path}")
    print(f"  Number of classes: {rf_model.n_classes_}")
    print(f"  Classes: {rf_model.classes_}")
    
except Exception as e:
    print(f"✗ Error loading RF model: {e}")
    traceback.print_exc()

# 4. Check XGB model
print("\n4. XGBOOST MODEL INSPECTION:")
print("-" * 70)
try:
    import joblib
    xgb_path = SOUND_MODEL_DIR / "xgb_model.pkl"
    xgb_model = joblib.load(xgb_path)
    print(f"✓ XGB model loaded from: {xgb_path}")
    print(f"  Number of classes: {xgb_model.n_classes_}")
    print(f"  Classes: {xgb_model.classes_}")
    
except Exception as e:
    print(f"✗ Error loading XGB model: {e}")
    traceback.print_exc()

# 5. Check feature orders
print("\n5. FEATURE ORDER INSPECTION:")
print("-" * 70)
try:
    feature_order_path = SOUND_MODEL_DIR / "feature_order.json"
    with open(feature_order_path, 'r') as f:
        feature_data = json.load(f)
        features = feature_data['features']
        print(f"✓ Feature order loaded: {len(features)} features")
        print(f"  First 5 features: {features[:5]}")
        print(f"  Last 5 features: {features[-5:]}")
except Exception as e:
    print(f"✗ Error loading feature order: {e}")
    traceback.print_exc()

# 6. Check hybrid feature order
print("\n6. HYBRID FEATURE ORDER INSPECTION:")
print("-" * 70)
try:
    hybrid_path = SOUND_MODEL_DIR / "hybrid_feature_order.json"
    with open(hybrid_path, 'r') as f:
        hybrid_data = json.load(f)
        hybrid_features = hybrid_data['features']
        print(f"✓ Hybrid feature order loaded: {len(hybrid_features)} features")
        print(f"  First 5 features: {hybrid_features[:5]}")
except Exception as e:
    print(f"✗ Error loading hybrid feature order: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
