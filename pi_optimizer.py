"""
Raspberry Pi 5 Optimization Utilities
Performance optimization techniques for running ML models on Raspberry Pi 5
"""

import os
import sys
import logging
import subprocess
from typing import Optional, Tuple
import threading
import time

logger = logging.getLogger(__name__)


class RaspberryPiOptimizer:
    """
    Optimization utilities for Raspberry Pi 5
    Handles memory management, CPU affinity, and performance tuning
    """
    
    def __init__(self):
        self.is_raspberry_pi = self._detect_raspberry_pi()
        if self.is_raspberry_pi:
            logger.info("Running on Raspberry Pi - applying optimizations")
        else:
            logger.info("Not running on Raspberry Pi - skipping Pi-specific optimizations")
    
    def _detect_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo
        except:
            return False
    
    def optimize_tensorflow(self):
        """Apply TensorFlow optimizations for Raspberry Pi"""
        # Set TensorFlow to use limited threads
        os.environ['TF_NUM_INTEROP_THREADS'] = '2'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
        
        # Disable TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Enable memory growth
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
        
        logger.info("TensorFlow optimizations applied")
    
    def optimize_opencv(self):
        """Apply OpenCV optimizations"""
        import cv2
        
        # Set number of threads for OpenCV
        cv2.setNumThreads(4)
        
        # Enable optimized code
        cv2.setUseOptimized(True)
        
        logger.info(f"OpenCV optimizations applied (threads: 4, optimized: {cv2.useOptimized()})")
    
    def optimize_numpy(self):
        """Apply NumPy optimizations"""
        import numpy as np
        
        # Limit NumPy threads
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        os.environ['NUMEXPR_NUM_THREADS'] = '4'
        
        logger.info("NumPy optimizations applied")
    
    def set_cpu_governor_performance(self) -> bool:
        """Set CPU governor to performance mode"""
        if not self.is_raspberry_pi:
            return False
        
        try:
            for i in range(4):  # Pi 5 has 4 cores
                subprocess.run(
                    ['sudo', 'cpufreq-set', '-c', str(i), '-g', 'performance'],
                    check=True, capture_output=True
                )
            logger.info("CPU governor set to performance mode")
            return True
        except Exception as e:
            logger.warning(f"Could not set CPU governor: {e}")
            return False
    
    def get_system_info(self) -> dict:
        """Get system information"""
        info = {
            'is_raspberry_pi': self.is_raspberry_pi,
            'cpu_count': os.cpu_count(),
            'platform': sys.platform
        }
        
        if self.is_raspberry_pi:
            try:
                # Get temperature
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000
                    info['cpu_temperature'] = temp
                
                # Get memory info
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal' in line:
                            info['total_memory_mb'] = int(line.split()[1]) // 1024
                        elif 'MemAvailable' in line:
                            info['available_memory_mb'] = int(line.split()[1]) // 1024
            except:
                pass
        
        return info
    
    def apply_all_optimizations(self):
        """Apply all available optimizations"""
        self.optimize_numpy()
        self.optimize_tensorflow()
        self.optimize_opencv()
        
        if self.is_raspberry_pi:
            self.set_cpu_governor_performance()
        
        logger.info("All optimizations applied")
        return self.get_system_info()


class MemoryMonitor:
    """Monitor memory usage during processing"""
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._running = False
        self._thread = None
    
    def start(self):
        """Start memory monitoring"""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop memory monitoring"""
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _monitor_loop(self):
        """Memory monitoring loop"""
        try:
            import psutil
            while self._running:
                mem = psutil.virtual_memory()
                if mem.percent > 85:
                    logger.warning(f"High memory usage: {mem.percent}%")
                time.sleep(self.interval)
        except ImportError:
            logger.warning("psutil not available - memory monitoring disabled")


class FrameRateController:
    """Control frame rate for consistent processing"""
    
    def __init__(self, target_fps: float = 10.0):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_time = None
    
    def wait(self):
        """Wait to maintain target frame rate"""
        current_time = time.time()
        
        if self.last_time is not None:
            elapsed = current_time - self.last_time
            sleep_time = self.frame_time - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.last_time = time.time()
    
    def get_actual_fps(self) -> Optional[float]:
        """Get actual achieved FPS"""
        if self.last_time is None:
            return None
        
        current_time = time.time()
        elapsed = current_time - self.last_time
        return 1.0 / elapsed if elapsed > 0 else self.target_fps


def warmup_models(pipeline):
    """
    Warmup models by running a dummy inference
    This helps with JIT compilation and cache warming
    """
    import numpy as np
    logger.info("Warming up models...")
    
    # Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Warmup detection
    _ = pipeline.elephant_detector.detect(dummy_frame)
    
    # Warmup pose classifier
    if pipeline.pose_classifier:
        _ = pipeline.pose_classifier.classify(dummy_frame)
    
    # Warmup sound classifier with dummy audio
    if pipeline.sound_classifier:
        dummy_audio = np.zeros(22050 * 3)  # 3 seconds
        _ = pipeline.sound_classifier.classify(dummy_audio)
    
    logger.info("Model warmup complete")


def check_dependencies() -> Tuple[bool, list]:
    """
    Check if all required dependencies are installed
    Returns (success, missing_packages)
    """
    required = [
        'cv2',
        'numpy',
        'tensorflow',
        'ultralytics',
        'librosa',
        'joblib',
        'sklearn'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            # Map package name to pip name
            pip_names = {
                'cv2': 'opencv-python',
                'sklearn': 'scikit-learn'
            }
            missing.append(pip_names.get(package, package))
    
    return len(missing) == 0, missing


if __name__ == "__main__":
    # Test optimizations
    logging.basicConfig(level=logging.INFO)
    
    optimizer = RaspberryPiOptimizer()
    info = optimizer.apply_all_optimizations()
    
    print("\nSystem Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Check dependencies
    success, missing = check_dependencies()
    if success:
        print("\nAll dependencies installed correctly!")
    else:
        print(f"\nMissing packages: {', '.join(missing)}")
