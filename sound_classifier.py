"""
Sound-Based Behavior Classification Module
Analyzes audio from video to classify elephant behavior as aggressive or calm
Uses CNN-LSTM deep learning model and ensemble of ML models
"""

import numpy as np
import json
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
import soundfile as sf

# Deep learning
import tensorflow as tf
from tensorflow import keras

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
    BEHAVIOR_LABEL_MAPPING,
    normalize_behavior_label,   # FIX: use shared normalizer from config
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SoundResult:
    """
    Data class for sound-based behavior classification result.
    Standard behavior values: "Aggressive", "Calm", or None (if prediction failed).
    """
    behavior: Optional[str]   # "Aggressive", "Calm", or None when prediction failed
    confidence: float
    # --- structured result fields for safe fusion in the pipeline ---
    valid: bool = True          # FIX: False when classifier could not produce a result
    error: Optional[str] = None # FIX: human-readable reason when valid=False
    source: str = "sound"       # always "sound" for this classifier
    # --- per-model detail (optional diagnostics) ---
    cnn_lstm_prediction: Optional[str] = None
    cnn_lstm_confidence: Optional[float] = None
    ml_prediction: Optional[str] = None
    ml_confidence: Optional[float] = None
    features: Optional[Dict] = None


class SoundBehaviorClassifier:
    """
    Sound-based behavior classifier using CNN-LSTM and ML ensemble
    Analyzes audio features to classify elephant vocalizations
    """
    
    def __init__(self):
        """
        Initialize the sound-based behavior classifier
        """
        logger.info("Loading sound classification models...")
        
        # Load CNN-LSTM model
        logger.info(f"Loading CNN-LSTM model from {SOUND_CNN_LSTM_MODEL}")
        self.cnn_lstm_model = keras.models.load_model(str(SOUND_CNN_LSTM_MODEL))
        
        # Load ML models
        logger.info(f"Loading RF model from {SOUND_RF_MODEL}")
        self.rf_model = joblib.load(str(SOUND_RF_MODEL))
        
        logger.info(f"Loading XGB model from {SOUND_XGB_MODEL}")
        self.xgb_model = joblib.load(str(SOUND_XGB_MODEL))
        
        # Load preprocessing components
        self.scaler = joblib.load(str(SOUND_SCALER))
        self.hybrid_scaler = joblib.load(str(SOUND_HYBRID_SCALER))
        self.label_encoder = joblib.load(str(SOUND_LABEL_ENCODER))
        
        # Load feature orders
        with open(str(SOUND_FEATURE_ORDER), 'r') as f:
            self.feature_order = json.load(f)['features']
        
        with open(str(SOUND_HYBRID_FEATURE_ORDER), 'r') as f:
            self.hybrid_feature_order = json.load(f)['features']
        
        logger.info("Sound behavior classifier initialized successfully")
    
    def classify_from_file(self, audio_path: str) -> SoundResult:
        """
        Classify behavior from an audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            SoundResult with behavior classification
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
            return self.classify(audio, sr)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            # FIX: return structured failure so pipeline can apply fallback logic
            return SoundResult(
                behavior=None,
                confidence=0.0,
                valid=False,
                error=f"Audio load failed: {e}"
            )
    
    def classify(self, audio: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> SoundResult:
        """
        Classify behavior from audio data
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate
            
        Returns:
            SoundResult with behavior classification
        """
        try:
            # Extract features
            features = self._extract_features(audio, sr)
            
            if features is None:
                # FIX: structured failure — pipeline will use pose-only fallback
                return SoundResult(
                    behavior=None,
                    confidence=0.0,
                    valid=False,
                    error="Feature extraction returned None"
                )

            # CNN-LSTM prediction (returns (label, conf) or (None, 0.0) on failure)
            cnn_lstm_result = self._predict_cnn_lstm(audio, sr)

            # ML ensemble prediction
            ml_result = self._predict_ml_ensemble(features)

            # Combine predictions (weighted average)
            final_behavior, final_confidence = self._combine_predictions(
                cnn_lstm_result, ml_result
            )

            # Determine validity: at least one sub-model must have produced a result
            is_valid = final_behavior is not None

            return SoundResult(
                behavior=final_behavior,
                confidence=final_confidence,
                valid=is_valid,
                error=None if is_valid else "Both CNN-LSTM and ML ensemble failed",
                cnn_lstm_prediction=cnn_lstm_result[0],
                cnn_lstm_confidence=cnn_lstm_result[1],
                ml_prediction=ml_result[0],
                ml_confidence=ml_result[1],
                features=features
            )

        except Exception as e:
            logger.error(f"Error classifying audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # FIX: never crash — return structured failure
            return SoundResult(
                behavior=None,
                confidence=0.0,
                valid=False,
                error=f"classify() exception: {e}"
            )
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> Optional[Dict]:
        """
        Extract audio features for ML classification
        """
        try:
            features = {}
            
            # Basic features
            features['rms'] = float(np.mean(librosa.feature.rms(y=audio)))
            features['zcr'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            # Dominant frequency
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            dominant_idx = np.argmax(magnitude[:len(magnitude)//2])
            features['dominant_freq'] = float(abs(freqs[dominant_idx]))
            features['dominant_frequency'] = features['dominant_freq']
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=AUDIO_N_MFCC)
            for i in range(AUDIO_N_MFCC):
                features[f'mfcc_{i+1}'] = float(np.mean(mfccs[i]))
                features[f'mfcc_std_{i+1}'] = float(np.std(mfccs[i]))
            
            # Delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs)
            for i in range(AUDIO_N_MFCC):
                features[f'delta_{i+1}'] = float(np.mean(delta_mfccs[i]))
            
            # Delta-delta MFCCs
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            for i in range(AUDIO_N_MFCC):
                features[f'delta2_{i+1}'] = float(np.mean(delta2_mfccs[i]))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma'] = float(np.mean(chroma))
            
            # Harmonic-to-noise ratio estimate
            harmonic, percussive = librosa.effects.hpss(audio)
            features['hnr'] = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6))
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            features['num_onsets'] = len(onset_frames)
            
            # Check for overlapping calls (multiple peaks in onset strength)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            peaks = np.where(onset_env > np.mean(onset_env) + np.std(onset_env))[0]
            features['overlapping_calls'] = 1 if len(peaks) > 2 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _predict_cnn_lstm(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
        """
        Predict using CNN-LSTM model
        """
        try:
            # FIX: guard against empty / invalid audio
            if audio is None or not hasattr(audio, '__len__') or len(audio) == 0:
                logger.warning("Empty audio input for CNN-LSTM prediction")
                return None, 0.0
            
            # Prepare mel spectrogram for CNN-LSTM
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr,
                n_fft=AUDIO_N_FFT,
                hop_length=AUDIO_HOP_LENGTH,
                n_mels=128
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
            
            # Reshape for model input
            # Expected shape: (batch, time_steps, features) or (batch, height, width, channels)
            input_shape = self.cnn_lstm_model.input_shape
            
            if len(input_shape) == 4:
                # CNN expects (batch, height, width, channels)
                mel_spec_resized = self._resize_spectrogram(mel_spec_db, input_shape[1], input_shape[2])
                mel_input = mel_spec_resized.reshape(1, input_shape[1], input_shape[2], 1)
            else:
                # LSTM expects (batch, time_steps, features)
                time_steps = input_shape[1] if input_shape[1] else mel_spec_db.shape[1]
                features = input_shape[2] if input_shape[2] else mel_spec_db.shape[0]
                mel_spec_resized = self._resize_spectrogram(mel_spec_db, features, time_steps)
                mel_input = mel_spec_resized.T.reshape(1, time_steps, features)
            
            # Predict
            prediction = self.cnn_lstm_model.predict(mel_input, verbose=0)
            
            # FIX: defensive check for empty / None prediction
            if prediction is None or (hasattr(prediction, 'size') and prediction.size == 0):
                logger.warning("Empty prediction from CNN-LSTM model")
                return None, 0.0
            
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
                    # Single output (binary sigmoid)
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
            
            # FIX: validate pred_class against ACTUAL number of label-encoder classes
            n_classes = len(self.label_encoder.classes_)
            if prediction.shape[-1] not in (1, n_classes):
                logger.warning(
                    f"CNN-LSTM output size {prediction.shape[-1]} does not match "
                    f"label encoder classes ({n_classes}). Using argmax fallback."
                )
            if pred_class >= n_classes or pred_class < 0:
                logger.warning(f"CNN-LSTM pred_class {pred_class} out of range [0,{n_classes}), defaulting to 0")
                pred_class = 0

            # Get behavior label from encoder and normalize to standard label
            raw_label = self.label_encoder.inverse_transform([pred_class])[0]
            behavior_mapped = normalize_behavior_label(raw_label)  # FIX: "Normal" -> "Calm"

            return behavior_mapped, confidence

        except Exception as e:
            logger.error(f"Error in CNN-LSTM prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, 0.0
    
    def _resize_spectrogram(self, spec: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """
        Resize spectrogram to target dimensions
        """
        import cv2
        resized = cv2.resize(spec, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def _predict_ml_ensemble(self, features: Dict) -> Tuple[str, float]:
        """
        Predict using ML ensemble (RF + XGB)
        """
        try:
            # FIX: guard against empty features dict
            if not features or len(features) == 0:
                logger.warning("Empty features for ML ensemble prediction")
                return None, 0.0
            
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
            
            # Random Forest prediction
            rf_pred = self.rf_model.predict(feature_scaled)[0]
            rf_proba = self.rf_model.predict_proba(feature_scaled)[0]
            rf_conf = float(np.max(rf_proba))

            # XGBoost prediction
            xgb_pred = self.xgb_model.predict(feature_scaled)[0]
            xgb_proba = self.xgb_model.predict_proba(feature_scaled)[0]
            xgb_conf = float(np.max(xgb_proba))

            # Weighted ensemble: higher-confidence model wins
            if rf_conf >= xgb_conf:
                final_pred = rf_pred
                final_conf = rf_conf * 0.6 + xgb_conf * 0.4
            else:
                final_pred = xgb_pred
                final_conf = xgb_conf * 0.6 + rf_conf * 0.4

            # FIX: ML models may return STRING labels (e.g. "Normal", "Aggressive")
            # instead of integer indices when they were trained with string targets.
            # We handle BOTH cases here so neither crashes.
            if isinstance(final_pred, str):
                # Model stored string labels — normalize directly ("Normal" -> "Calm")
                behavior_mapped = normalize_behavior_label(final_pred)
            else:
                # Model stored integer indices — decode via label encoder
                idx = int(final_pred)
                n_classes = len(self.label_encoder.classes_)
                if idx < 0 or idx >= n_classes:
                    logger.warning(f"ML pred index {idx} out of range [0,{n_classes}), defaulting to 0")
                    idx = 0
                raw_label = self.label_encoder.inverse_transform([idx])[0]
                behavior_mapped = normalize_behavior_label(raw_label)  # FIX: "Normal" -> "Calm"

            return behavior_mapped, final_conf

        except Exception as e:
            logger.error(f"Error in ML ensemble prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, 0.0
    
    def _normalize_behavior_label(self, label: str) -> str:
        """
        Instance-level wrapper around the module-level normalize_behavior_label().
        Kept for backward compatibility; delegates all logic to config.py.
        Returns proper-case standard labels: "Aggressive", "Calm", or the raw
        stripped label when the value is not recognised.
        """
        return normalize_behavior_label(label)
    
    def _combine_predictions(self, cnn_lstm_result: Tuple[str, float], 
                            ml_result: Tuple[str, float]) -> Tuple[str, float]:
        """
        Combine CNN-LSTM and ML ensemble predictions
        Uses weighted voting based on confidence
        """
        cnn_behavior, cnn_conf = cnn_lstm_result
        ml_behavior, ml_conf = ml_result

        # FIX: normalize both inputs; also handle None (failed sub-model)
        cnn_behavior = normalize_behavior_label(cnn_behavior) if cnn_behavior else None
        ml_behavior = normalize_behavior_label(ml_behavior) if ml_behavior else None
        
        # Weight: CNN-LSTM 0.6, ML 0.4 (deep learning typically more reliable for audio)
        cnn_weight = 0.6
        ml_weight = 0.4

        # FIX: handle cases where one or both sub-models returned None (failure)
        if cnn_behavior is None and ml_behavior is None:
            return None, 0.0
        if cnn_behavior is None:
            return ml_behavior, ml_conf * ml_weight
        if ml_behavior is None:
            return cnn_behavior, cnn_conf * cnn_weight

        if cnn_behavior == ml_behavior:
            # Agreement — boost confidence slightly
            final_behavior = cnn_behavior
            final_confidence = cnn_conf * cnn_weight + ml_conf * ml_weight
        else:
            # Disagreement — choose the more confident sub-model
            if cnn_conf * cnn_weight >= ml_conf * ml_weight:
                final_behavior = cnn_behavior
                final_confidence = cnn_conf * cnn_weight
            else:
                final_behavior = ml_behavior
                final_confidence = ml_conf * ml_weight
        
        return final_behavior, final_confidence


def extract_audio_from_video(video_path: str, output_path: str = None) -> Optional[str]:
    """
    Extract audio from video file
    
    Args:
        video_path: Path to video file
        output_path: Path for output audio file (optional)
        
    Returns:
        Path to extracted audio file
    """
    import subprocess
    import tempfile
    import os
    
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.wav')
    
    try:
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM format
            '-ar', str(AUDIO_SAMPLE_RATE),  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")
        return None
    except FileNotFoundError:
        logger.warning("ffmpeg not found. Trying moviepy...")
        try:
            from moviepy.editor import VideoFileClip
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(output_path, fps=AUDIO_SAMPLE_RATE, verbose=False, logger=None)
            return output_path
        except Exception as e:
            logger.error(f"Error extracting audio with moviepy: {e}")
            return None


if __name__ == "__main__":
    import sys
    
    classifier = SoundBehaviorClassifier()
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        result = classifier.classify_from_file(audio_path)
        print(f"Behavior: {result.behavior}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"CNN-LSTM: {result.cnn_lstm_prediction} ({result.cnn_lstm_confidence:.2f})")
        print(f"ML Ensemble: {result.ml_prediction} ({result.ml_confidence:.2f})")
    else:
        print("Sound Behavior Classifier initialized successfully.")
        print("Usage: python sound_classifier.py <audio_path>")
