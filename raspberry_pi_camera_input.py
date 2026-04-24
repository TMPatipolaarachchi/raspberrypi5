#!/usr/bin/env python3
"""
Raspberry Pi Camera Module V2 — Real-Time Input for Elephant Detection Pipeline
================================================================================

This module provides an optional real-time camera input source using the
Raspberry Pi Camera Module V2 connected via the CSI port.

VIDEO FILE UPLOAD remains the primary (main) mode.
Camera mode is OPTIONAL and must be explicitly enabled.

Library priority
----------------
1. picamera2  — native Raspberry Pi 5 / libcamera backend (preferred)
2. OpenCV     — fallback using /dev/video0 V4L2 device

Frame output
------------
All frames are returned as BGR numpy arrays (H x W x 3, dtype=uint8)
compatible with OpenCV and YOLO / TensorFlow model inputs:

    frame = camera.get_frame()
    result = elephant_detector.detect(frame)       # works directly

Integration
-----------
Pipeline integration example (in main.py / integrated_pipeline.py):

    if mode == "video":
        pipeline.process_video(video_path)
    elif mode == "camera":
        for frame in camera_frame_generator():
            detection = pipeline.elephant_detector.detect(frame, annotate=False)
            ...
"""

import cv2
import time
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Iterator, Tuple

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "camera.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("PiCamera")

# ── Camera configuration constants ───────────────────────────────────────────
# Resolution suitable for YOLO inference on Raspberry Pi 5.
# 640×480 is recommended for best balance of speed and accuracy.
# Change to (1280, 720) for 720p if the Pi 5 load permits.
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 20          # 15–30 FPS; 20 is a safe default
CAMERA_INDEX  = 0           # V4L2 device index (/dev/video0) for OpenCV fallback

# Process every Nth frame to reduce CPU load.
# Set to 1 to process every frame; increase to reduce load.
FRAME_SKIP_CAMERA = 3

# Seconds to wait for the camera to warm up after initialization.
CAMERA_WARMUP_SECONDS = 2.0

# ── Backend availability flags ────────────────────────────────────────────────
# These are resolved once at import time so the rest of the module can branch
# cleanly without repeated try/except blocks.
_PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2             # type: ignore
    _PICAMERA2_AVAILABLE = True
    logger.info("picamera2 backend available")
except ImportError:
    logger.info("picamera2 not installed — will use OpenCV/V4L2 fallback")


# ══════════════════════════════════════════════════════════════════════════════
#  PiCameraStream  — unified camera wrapper
# ══════════════════════════════════════════════════════════════════════════════

class PiCameraStream:
    """
    Wrapper around Raspberry Pi Camera Module V2.

    Tries picamera2 first (recommended for RPi 5 / libcamera).
    Falls back to OpenCV VideoCapture (/dev/video0) automatically.

    Usage
    -----
        cam = PiCameraStream()
        cam.initialize_camera()
        cam.start_camera_stream()

        frame = cam.get_frame()          # single frame

        for frame in cam.camera_frame_generator():
            ...                          # continuous frames

        cam.stop_camera_stream()
    """

    def __init__(
        self,
        width:  int   = CAMERA_WIDTH,
        height: int   = CAMERA_HEIGHT,
        fps:    int   = CAMERA_FPS,
        frame_skip: int = FRAME_SKIP_CAMERA,
    ):
        self.width      = width
        self.height     = height
        self.fps        = fps
        self.frame_skip = frame_skip

        # Internal state
        self._backend: str             = "none"   # "picamera2" | "opencv"
        self._picam: Optional[object]  = None     # Picamera2 instance
        self._cap:   Optional[cv2.VideoCapture] = None  # OpenCV cap
        self._running: bool            = False
        self._lock    = threading.Lock()

    # ── 1. initialize_camera() ────────────────────────────────────────────────

    def initialize_camera(self) -> bool:
        """
        Initialize the Raspberry Pi Camera Module V2.

        Tries picamera2 first, then falls back to OpenCV.

        Returns
        -------
        bool
            True if camera was initialized successfully, False otherwise.
        """
        logger.info("Initializing Raspberry Pi Camera Module V2 ...")

        if _PICAMERA2_AVAILABLE:
            success = self._init_picamera2()
            if success:
                return True
            logger.warning("picamera2 init failed — trying OpenCV fallback")

        return self._init_opencv()

    def _init_picamera2(self) -> bool:
        """Initialize using picamera2 / libcamera backend."""
        try:
            picam = Picamera2()

            # Build a preview (still_capable) config at our target resolution
            config = picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            picam.configure(config)

            # Set frame rate via frame-duration limits (microseconds)
            frame_duration_us = int(1_000_000 / self.fps)
            picam.set_controls({
                "FrameDurationLimits": (frame_duration_us, frame_duration_us)
            })

            self._picam    = picam
            self._backend  = "picamera2"
            logger.info(
                f"picamera2 initialized: {self.width}x{self.height} @ {self.fps} FPS"
            )
            return True

        except Exception as exc:
            logger.error(f"picamera2 initialization error: {exc}")
            self._picam = None
            return False

    def _init_opencv(self) -> bool:
        """Initialize using OpenCV VideoCapture (V4L2 / /dev/video0)."""
        try:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                logger.error(
                    f"OpenCV could not open camera device {CAMERA_INDEX}. "
                    "Check if the camera is connected and /dev/video0 exists."
                )
                return False

            # Apply desired settings; the driver may round to nearest supported value
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS,          self.fps)

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_f = cap.get(cv2.CAP_PROP_FPS)

            self._cap     = cap
            self._backend = "opencv"
            logger.info(
                f"OpenCV camera initialized: {actual_w}x{actual_h} @ {actual_f:.1f} FPS "
                f"(requested {self.width}x{self.height} @ {self.fps} FPS)"
            )
            return True

        except Exception as exc:
            logger.error(f"OpenCV camera initialization error: {exc}")
            return False

    # ── 2. start_camera_stream() ──────────────────────────────────────────────

    def start_camera_stream(self) -> bool:
        """
        Start the camera capture stream.

        Must be called after initialize_camera().

        Returns
        -------
        bool
            True if streaming started, False on error.
        """
        if self._running:
            logger.warning("Camera stream is already running")
            return True

        if self._backend == "none":
            logger.error("Camera not initialized. Call initialize_camera() first.")
            return False

        try:
            if self._backend == "picamera2" and self._picam is not None:
                self._picam.start()
                # Allow auto-exposure / auto-white-balance to settle
                time.sleep(CAMERA_WARMUP_SECONDS)

            # OpenCV VideoCapture starts implicitly when opened;
            # just wait for the sensor to stabilize.
            if self._backend == "opencv":
                time.sleep(CAMERA_WARMUP_SECONDS)

            self._running = True
            logger.info(f"Camera stream started (backend: {self._backend})")
            return True

        except Exception as exc:
            logger.error(f"Failed to start camera stream: {exc}")
            return False

    # ── 3. get_frame() ────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        Returns
        -------
        np.ndarray | None
            BGR frame (H x W x 3, uint8) compatible with OpenCV and YOLO, or
            None if capture failed.

        Example
        -------
            frame = cam.get_frame()
            if frame is not None:
                result = elephant_detector.detect(frame)
        """
        if not self._running:
            logger.error("Camera stream is not running. Call start_camera_stream() first.")
            return None

        with self._lock:
            try:
                if self._backend == "picamera2" and self._picam is not None:
                    return self._frame_from_picamera2()

                if self._backend == "opencv" and self._cap is not None:
                    return self._frame_from_opencv()

                logger.error("No active camera backend.")
                return None

            except Exception as exc:
                logger.error(f"Frame capture error: {exc}")
                return None

    def _frame_from_picamera2(self) -> Optional[np.ndarray]:
        """Capture one frame via picamera2 and convert RGB→BGR."""
        rgb_frame = self._picam.capture_array()   # returns (H, W, 3) RGB uint8
        if rgb_frame is None:
            logger.warning("picamera2 returned None frame")
            return None
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        logger.debug("Frame captured via picamera2")
        return bgr_frame

    def _frame_from_opencv(self) -> Optional[np.ndarray]:
        """Read one frame via OpenCV VideoCapture."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("OpenCV read() returned no frame")
            return None
        logger.debug("Frame captured via OpenCV")
        return frame   # OpenCV already returns BGR

    # ── 4. stop_camera_stream() ───────────────────────────────────────────────

    def stop_camera_stream(self) -> None:
        """
        Safely stop the camera stream and release all resources.

        Safe to call even if the camera was never started.
        """
        logger.info("Stopping camera stream ...")
        self._running = False

        with self._lock:
            if self._backend == "picamera2" and self._picam is not None:
                try:
                    self._picam.stop()
                    self._picam.close()
                    logger.info("picamera2 resources released")
                except Exception as exc:
                    logger.warning(f"Error closing picamera2: {exc}")
                finally:
                    self._picam = None

            if self._backend == "opencv" and self._cap is not None:
                try:
                    self._cap.release()
                    logger.info("OpenCV VideoCapture released")
                except Exception as exc:
                    logger.warning(f"Error releasing OpenCV capture: {exc}")
                finally:
                    self._cap = None

        self._backend = "none"
        logger.info("Camera stream stopped.")

    # ── 5. camera_frame_generator() ──────────────────────────────────────────

    def camera_frame_generator(
        self,
        max_frames: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """
        Generator that continuously yields camera frames.

        Applies frame skipping to reduce CPU load.

        Parameters
        ----------
        max_frames : int | None
            Stop after this many *yielded* frames (None = run until stopped).
        skip : int | None
            Override the instance frame_skip value for this run.

        Yields
        ------
        np.ndarray
            BGR frame (H x W x 3, uint8).

        Usage in pipeline
        -----------------
            for frame in cam.camera_frame_generator():
                detection = pipeline.elephant_detector.detect(frame, annotate=False)
                ...
        """
        effective_skip = skip if skip is not None else self.frame_skip
        yielded  = 0
        captured = 0

        logger.info(
            f"Frame generator started — skip every {effective_skip} frame(s), "
            f"max_frames={'unlimited' if max_frames is None else max_frames}"
        )

        while self._running:
            captured += 1

            # Apply frame skipping: only process every Nth frame.
            if captured % effective_skip != 0:
                # Still consume the frame to keep the camera buffer moving.
                _ = self.get_frame()
                continue

            frame = self.get_frame()
            if frame is None:
                logger.warning("Skipping None frame from generator")
                time.sleep(0.05)   # brief pause before retry
                continue

            yield frame
            yielded += 1

            if max_frames is not None and yielded >= max_frames:
                logger.info(f"Reached max_frames limit ({max_frames}) — stopping generator")
                break

        logger.info(
            f"Frame generator stopped — captured: {captured}, yielded: {yielded}"
        )

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True if the camera stream is currently active."""
        return self._running

    @property
    def backend(self) -> str:
        """Active backend name: 'picamera2', 'opencv', or 'none'."""
        return self._backend

    @property
    def resolution(self) -> Tuple[int, int]:
        """Configured (width, height) resolution."""
        return (self.width, self.height)

    def __repr__(self) -> str:
        return (
            f"PiCameraStream(backend={self._backend!r}, "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps}, running={self._running})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Module-level convenience functions
#  These allow simple procedural usage without instantiating the class.
# ══════════════════════════════════════════════════════════════════════════════

def initialize_camera(
    width:  int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    fps:    int = CAMERA_FPS,
) -> Optional[PiCameraStream]:
    """
    Initialize and return a ready-to-use PiCameraStream.

    Returns None if initialization fails.

    Example
    -------
        cam = initialize_camera()
        if cam:
            start_camera_stream(cam)
    """
    cam = PiCameraStream(width=width, height=height, fps=fps)
    success = cam.initialize_camera()
    if not success:
        logger.error("initialize_camera() failed — returning None")
        return None
    return cam


def start_camera_stream(cam: PiCameraStream) -> bool:
    """Start the stream on an already-initialized PiCameraStream."""
    return cam.start_camera_stream()


def get_frame(cam: PiCameraStream) -> Optional[np.ndarray]:
    """Capture one BGR frame from a running PiCameraStream."""
    return cam.get_frame()


def stop_camera_stream(cam: PiCameraStream) -> None:
    """Stop the stream and release resources."""
    cam.stop_camera_stream()


def camera_frame_generator(
    cam: PiCameraStream,
    max_frames: Optional[int] = None,
    skip: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Yield BGR frames continuously from a running PiCameraStream.

    This is the primary integration point for integrated_pipeline.py:

        cam = initialize_camera()
        start_camera_stream(cam)
        for frame in camera_frame_generator(cam):
            detection = elephant_detector.detect(frame, annotate=False)
        stop_camera_stream(cam)
    """
    yield from cam.camera_frame_generator(max_frames=max_frames, skip=skip)


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test / Example usage  (python raspberry_pi_camera_input.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Raspberry Pi Camera Module V2 — Self-Test")
    print("=" * 60)

    # ── Example 1 — Camera preview test ──────────────────────────────────────
    print("\n[Example 1] Camera preview — press 'q' to stop\n")

    cam = PiCameraStream(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        frame_skip=1,   # no skipping for the display demo
    )

    if not cam.initialize_camera():
        print(
            "ERROR: Could not initialize camera.\n"
            "  • Check that the Camera Module V2 is connected to the CSI port.\n"
            "  • Run:  sudo raspi-config  → Interface Options → Camera → Enable\n"
            "  • On RPi 5 with picamera2:  sudo apt-get install python3-picamera2\n"
            "  • Reboot after enabling.\n"
        )
        sys.exit(1)

    if not cam.start_camera_stream():
        print("ERROR: Could not start camera stream.")
        sys.exit(1)

    print(f"Camera active: {cam}")
    print("Streaming — press 'q' in the preview window to stop ...\n")

    frame_count = 0
    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                print("Warning: received None frame, retrying ...")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Overlay frame counter
            cv2.putText(
                frame,
                f"Frame: {frame_count}  |  {cam.width}x{cam.height}  |  {cam.backend}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Pi Camera V2 — Preview (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User pressed 'q' — stopping.")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — stopping camera.")
    finally:
        cam.stop_camera_stream()
        cv2.destroyAllWindows()
        print(f"Example 1 done — {frame_count} frames captured.\n")

    # ── Example 2 — Pipeline integration simulation ───────────────────────────
    print("=" * 60)
    print("[Example 2] Pipeline integration simulation (10 frames)\n")

    cam2 = PiCameraStream(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        frame_skip=FRAME_SKIP_CAMERA,
    )

    if cam2.initialize_camera() and cam2.start_camera_stream():
        print(f"Camera ready: {cam2}")
        print("Simulating elephant_detector.detect(frame) ...\n")

        # In production, replace this with the real pipeline:
        #   from elephant_detector import ElephantDetector
        #   detector = ElephantDetector()
        #   for frame in cam2.camera_frame_generator(max_frames=10):
        #       result = detector.detect(frame, annotate=False)

        for idx, frame in enumerate(cam2.camera_frame_generator(max_frames=10), start=1):
            h, w = frame.shape[:2]
            print(
                f"  Frame {idx:2d}: shape=({h}, {w}, {frame.shape[2]}), "
                f"dtype={frame.dtype}, "
                f"mean_px={frame.mean():.1f}"
            )

        cam2.stop_camera_stream()
        print("\nExample 2 done — camera stream closed cleanly.")
    else:
        print("Example 2 skipped — camera not available.")

    print("\n" + "=" * 60)
    print("Self-test complete.")
    print("=" * 60)
