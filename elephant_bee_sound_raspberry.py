"""
Elephant Bee Sound Deterrent Module
====================================
Plays bee buzzing sounds through a Bluetooth speaker connected to the
Raspberry Pi 5 to safely repel elephants when specific conditions are met.

Activation requires ALL of:
    - elephant_detected == True
    - behavior == "calm"
    - group_type in ("herd", "individual")
    - current time within allowed daily windows

Deactivation (never play) if ANY of:
    - elephant_detected == False
    - behavior == "aggressive"
    - group_type == "family"

Usage from main pipeline:
    from elephant_bee_sound_raspberry import process_bee_sound

    result = {
        "elephant_detected": True,
        "behavior": "calm",
        "group_type": "herd"
    }
    process_bee_sound(result)
"""

import os
import random
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
BEE_SOUNDS_DIR = BASE_DIR / "bee_sounds"

# Playback duration in minutes
PLAYBACK_DURATION_MINUTES = 10

# Daily time windows during which bee sound playback is permitted.
# Each tuple is (start_time, end_time) in 24-hour "HH:MM" format.
ALLOWED_PLAY_TIME = [
    ("18:00", "23:00"),
    ("04:30", "07:30"),
]

# Supported audio file extensions
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("bee_sound")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Module state — tracks the last played clip to avoid consecutive repeats
# ---------------------------------------------------------------------------

_last_played_clip: str | None = None


# ===========================================================================
# 1. is_within_allowed_time
# ===========================================================================

def is_within_allowed_time() -> bool:
    """
    Check whether the current system time falls inside any of the
    configured daily time windows in ``ALLOWED_PLAY_TIME``.

    Returns True if playback is permitted right now, False otherwise.
    """
    now = datetime.now().time()

    for start_str, end_str in ALLOWED_PLAY_TIME:
        start = datetime.strptime(start_str, "%H:%M").time()
        end = datetime.strptime(end_str, "%H:%M").time()

        if start <= end:
            # Normal window (e.g. 18:00 – 23:00)
            if start <= now <= end:
                return True
        else:
            # Overnight window (e.g. 23:00 – 04:30)
            if now >= start or now <= end:
                return True

    return False


# ===========================================================================
# 2. select_bee_sound
# ===========================================================================

def select_bee_sound() -> str | None:
    """
    Select a random bee sound file from the ``bee_sounds/`` directory.

    Rules:
        - Must not repeat the previously played clip consecutively.
        - Returns the absolute file path of the selected clip.
        - Returns None if no suitable clip is available.
    """
    global _last_played_clip

    if not BEE_SOUNDS_DIR.is_dir():
        logger.error("Bee sounds directory not found: %s", BEE_SOUNDS_DIR)
        return None

    # Collect all supported audio files
    clips = [
        str(f) for f in BEE_SOUNDS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not clips:
        logger.error("No audio files found in %s", BEE_SOUNDS_DIR)
        return None

    # Filter out the last played clip to avoid consecutive repeat
    if len(clips) > 1 and _last_played_clip in clips:
        candidates = [c for c in clips if c != _last_played_clip]
    else:
        candidates = clips

    selected = random.choice(candidates)
    _last_played_clip = selected

    logger.info("Selected bee sound clip: %s", os.path.basename(selected))
    return selected


# ===========================================================================
# 3. play_bee_sound
# ===========================================================================

def play_bee_sound(file_path: str, duration_minutes: int = PLAYBACK_DURATION_MINUTES) -> bool:
    """
    Play the given audio file through the system's default audio output
    (Bluetooth speaker) for the specified duration.

    Playback strategy (in order of preference, all lightweight):
        1. mpg123  — lightweight CLI player for MP3
        2. aplay   — ALSA player for WAV
        3. pygame  — Python library fallback

    The clip is looped as needed to fill the requested duration.

    Args:
        file_path:        Absolute path to the audio file.
        duration_minutes: How long to play (default 10 minutes).

    Returns:
        True if playback completed (or ran for the full duration),
        False if playback failed.
    """
    if not os.path.isfile(file_path):
        logger.error("Audio file does not exist: %s", file_path)
        return False

    duration_seconds = duration_minutes * 60
    file_ext = os.path.splitext(file_path)[1].lower()
    basename = os.path.basename(file_path)

    logger.info("Starting bee sound playback: %s for %d minutes", basename, duration_minutes)

    # ---- Try mpg123 (best for MP3 on Pi) ----
    if file_ext == ".mp3" and _command_available("mpg123"):
        return _play_with_loop_subprocess(
            ["mpg123", "-q", file_path], duration_seconds, basename
        )

    # ---- Try aplay (best for WAV on Pi) ----
    if file_ext == ".wav" and _command_available("aplay"):
        return _play_with_loop_subprocess(
            ["aplay", "-q", file_path], duration_seconds, basename
        )

    # ---- Fallback: pygame ----
    if _play_with_pygame(file_path, duration_seconds, basename):
        return True

    # ---- Last resort: try mpg123/aplay regardless of extension ----
    for cmd in ("mpg123", "aplay"):
        if _command_available(cmd):
            return _play_with_loop_subprocess(
                [cmd, "-q", file_path], duration_seconds, basename
            )

    logger.error("No suitable audio player found. Install mpg123, aplay, or pygame.")
    return False


def _command_available(name: str) -> bool:
    """Check whether *name* is available on PATH."""
    try:
        subprocess.run(
            ["which", name], capture_output=True, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # On Windows, 'which' doesn't exist — try 'where' as fallback
        try:
            subprocess.run(
                ["where", name], capture_output=True, check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


def _play_with_loop_subprocess(cmd: list, duration_seconds: int, label: str) -> bool:
    """
    Play audio by spawning *cmd* in a loop until *duration_seconds* elapses.
    Each invocation plays the clip once; it is restarted as needed to fill
    the full duration.
    """
    start = time.time()
    try:
        while time.time() - start < duration_seconds:
            remaining = duration_seconds - (time.time() - start)
            if remaining <= 0:
                break
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5)
                break

        elapsed = round(time.time() - start, 1)
        logger.info("Bee sound playback finished: %s (played %.1f s)", label, elapsed)
        return True

    except Exception as exc:
        logger.error("Subprocess playback failed for %s: %s", label, exc)
        return False


def _play_with_pygame(file_path: str, duration_seconds: int, label: str) -> bool:
    """Attempt playback using pygame.mixer (optional dependency)."""
    try:
        import pygame  # noqa: optional dependency
    except ImportError:
        logger.debug("pygame not available — skipping pygame playback")
        return False

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(loops=-1)  # loop indefinitely

        logger.info("Playing via pygame: %s", label)
        time.sleep(duration_seconds)

        pygame.mixer.music.stop()
        pygame.mixer.quit()

        logger.info("Bee sound playback finished: %s (played %d s)", label, duration_seconds)
        return True

    except Exception as exc:
        logger.error("pygame playback failed for %s: %s", label, exc)
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        return False


# ===========================================================================
# 4. should_activate_bee_sound
# ===========================================================================

def should_activate_bee_sound(result: dict) -> bool:
    """
    Evaluate the AI pipeline result and decide whether to activate
    the bee sound deterrent.

    Activation requires ALL of:
        - elephant_detected == True
        - behavior == "calm"
        - group_type in ("herd", "individual")

    Returns False (never activate) if:
        - elephant_detected is False / missing
        - behavior == "aggressive"
        - group_type == "family"
    """
    if not isinstance(result, dict):
        logger.warning("Invalid result type: %s", type(result))
        return False

    # --- elephant_detected ---
    elephant_detected = result.get("elephant_detected", False)
    if not elephant_detected:
        logger.info("Bee sound skipped: no elephant detected")
        return False

    # --- behavior ---
    behavior = str(result.get("behavior", "")).strip().lower()
    if behavior == "aggressive":
        logger.info("Bee sound skipped: behavior is aggressive")
        return False
    if behavior != "calm":
        logger.info("Bee sound skipped: behavior is '%s' (not 'calm')", behavior)
        return False

    # --- group_type ---
    group_type = str(result.get("group_type", "")).strip().lower()
    if group_type == "family":
        logger.info("Bee sound skipped: group_type is family")
        return False
    if group_type not in ("herd", "individual"):
        logger.info("Bee sound skipped: group_type '%s' not in (herd, individual)", group_type)
        return False

    logger.info(
        "Bee sound activation conditions met: elephant=%s, behavior=%s, group=%s",
        elephant_detected, behavior, group_type,
    )
    return True


# ===========================================================================
# 5. process_bee_sound  —  MAIN ENTRY POINT
# ===========================================================================

def process_bee_sound(result: dict) -> dict:
    """
    Main entry point — evaluate conditions, check time window, select a
    clip, and play the bee sound if everything passes.

    Args:
        result: Final AI pipeline detection result dictionary.

    Returns:
        Structured status dictionary:
        {
            "activated": bool,
            "reason": str,
            "clip": str or None,
            "played_minutes": int or 0
        }
    """
    # Step 1: Check activation conditions
    if not should_activate_bee_sound(result):
        return {
            "activated": False,
            "reason": "Activation conditions not met",
            "clip": None,
            "played_minutes": 0,
        }

    # Step 2: Check daily time restriction
    if not is_within_allowed_time():
        logger.info("Bee sound skipped: current time outside allowed window %s", ALLOWED_PLAY_TIME)
        return {
            "activated": False,
            "reason": "Outside allowed time window",
            "clip": None,
            "played_minutes": 0,
        }

    # Step 3: Select a bee sound clip
    clip = select_bee_sound()
    if clip is None:
        logger.warning("Bee sound skipped: no clip available")
        return {
            "activated": False,
            "reason": "No bee sound clip available",
            "clip": None,
            "played_minutes": 0,
        }

    # Step 4: Play the bee sound
    logger.info("Activating bee sound deterrent")
    success = play_bee_sound(clip, duration_minutes=PLAYBACK_DURATION_MINUTES)

    if success:
        logger.info("Bee sound deterrent cycle complete")
        return {
            "activated": True,
            "reason": "Bee sound played successfully",
            "clip": os.path.basename(clip),
            "played_minutes": PLAYBACK_DURATION_MINUTES,
        }

    logger.error("Bee sound playback failed")
    return {
        "activated": False,
        "reason": "Playback failed",
        "clip": os.path.basename(clip),
        "played_minutes": 0,
    }


# ===========================================================================
# Example usage / self-test
# ===========================================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Override playback duration to 3 seconds for quick testing
    PLAYBACK_DURATION_MINUTES = 0  # 0 minutes = skip actual playback in test

    def _test_conditions_only(label: str, result: dict):
        """Helper: test should_activate + time check without actual playback."""
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        print(f"  Input:  {json.dumps(result, indent=2)}")

        activate = should_activate_bee_sound(result)
        in_time = is_within_allowed_time()

        print(f"  should_activate_bee_sound -> {activate}")
        print(f"  is_within_allowed_time    -> {in_time}")

        if activate and in_time:
            clip = select_bee_sound()
            print(f"  Selected clip             -> {clip}")
            print(f"  RESULT: Bee sound WOULD ACTIVATE")
        elif activate and not in_time:
            print(f"  RESULT: Conditions met but OUTSIDE time window")
        else:
            print(f"  RESULT: Bee sound NOT activated")

    # Case 1: All conditions met — should activate
    _test_conditions_only("CASE 1: calm + herd -> SHOULD ACTIVATE", {
        "elephant_detected": True,
        "behavior": "calm",
        "group_type": "herd",
    })

    # Case 2: Aggressive behavior — must NOT activate
    _test_conditions_only("CASE 2: aggressive + herd -> NO ACTIVATE", {
        "elephant_detected": True,
        "behavior": "aggressive",
        "group_type": "herd",
    })

    # Case 3: Family group — must NOT activate
    _test_conditions_only("CASE 3: calm + family -> NO ACTIVATE", {
        "elephant_detected": True,
        "behavior": "calm",
        "group_type": "family",
    })

    # Case 4: No elephant — must NOT activate
    _test_conditions_only("CASE 4: no elephant -> NO ACTIVATE", {
        "elephant_detected": False,
    })

    # Case 5: Individual + calm — should activate
    _test_conditions_only("CASE 5: calm + individual -> SHOULD ACTIVATE", {
        "elephant_detected": True,
        "behavior": "calm",
        "group_type": "individual",
    })

    # Case 6: Empty dict — should not crash
    _test_conditions_only("CASE 6: empty dict -> NO ACTIVATE (safe)", {})

    print(f"\n{'=' * 60}")
    print("  All tests completed.")
    print(f"{'=' * 60}")
