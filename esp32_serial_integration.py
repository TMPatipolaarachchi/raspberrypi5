"""
ESP32 Serial Integration Module for Elephant Detection and Behavior Classification System
=========================================================================================
Sends validated detection results from Raspberry Pi 5 to an ESP32 over UART serial.

Communication: Serial UART only (/dev/serial0 at 115200 baud).
No LoRa. No MQTT.

Transmission rule:
    elephant_detected == True  → send full JSON payload to ESP32
    elephant_detected == False → send minimal payload (pillar_id + elephant=false)

Usage from the main pipeline:
    from esp32_serial_integration import process_and_send_to_esp32

    result = {
        "elephant_detected": True,
        "adult_count": 2,
        "calf_count": 1,
        "elephant_count": 3,
        "group_type": "family",
        "behavior": "calm",
        "timestamp": "2026-03-08T10:30:00",
    }
    status = process_and_send_to_esp32(result)
"""

import json
import time
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

PILLAR_ID = "PILLAR_01"
PILLAR_LATITUDE = 7.123456
PILLAR_LONGITUDE = 80.123456

SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2  # seconds

# ---------------------------------------------------------------------------
# Allowed label sets
# ---------------------------------------------------------------------------

ALLOWED_BEHAVIORS = {"aggressive", "calm"}
ALLOWED_GROUP_TYPES = {"individual", "family", "herd"}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("esp32_serial")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ===========================================================================
# 1. normalize_behavior_label
# ===========================================================================

def normalize_behavior_label(label: str) -> str:
    """
    Map any behavior label variant to one of exactly two allowed outputs:
        "aggressive" or "calm"

    Rules:
        Normal     -> calm
        Calm       -> calm
        Aggressive -> aggressive
        (anything else) -> calm
    """
    if not isinstance(label, str) or not label.strip():
        return "calm"

    cleaned = label.strip().lower()

    mapping = {
        "aggressive": "aggressive",
        "agitated": "aggressive",
        "threat": "aggressive",
        "threatening": "aggressive",
        "trumpet": "aggressive",
        "trumpeting": "aggressive",
        "charging": "aggressive",
        "attack": "aggressive",
        "calm": "calm",
        "normal": "calm",
        "relaxed": "calm",
        "passive": "calm",
        "feeding": "calm",
        "resting": "calm",
        "neutral": "calm",
    }
    return mapping.get(cleaned, "calm")

# ===========================================================================
# 2. normalize_group_type
# ===========================================================================

def normalize_group_type(
    group_type: str,
    adult_count: int,
    calf_count: int,
    elephant_count: int,
) -> str:
    """
    Determine the correct group classification based on actual counts.

    Rules:
        elephant_count == 1              -> "individual"
        adult_count == 2 and calves >= 1 -> "family"
        otherwise                        -> "herd"
    """
    adult_count = max(int(adult_count), 0) if adult_count is not None else 0
    calf_count = max(int(calf_count), 0) if calf_count is not None else 0
    elephant_count = max(int(elephant_count), 0) if elephant_count is not None else 0

    if elephant_count == 1:
        return "individual"
    if adult_count == 2 and calf_count >= 1:
        return "family"
    if elephant_count > 1:
        return "herd"

    # Fallback: trust the incoming label if counts don't resolve
    if isinstance(group_type, str) and group_type.strip().lower() in ALLOWED_GROUP_TYPES:
        return group_type.strip().lower()

    return "herd"

# ===========================================================================
# 3. validate_detection_result
# ===========================================================================

def validate_detection_result(result: dict) -> dict:
    """
    Validate and sanitize a raw pipeline result dictionary.

    - Fills missing fields with safe defaults.
    - Normalises behavior and group_type labels.
    - Ensures counts are consistent non-negative integers.
    - Never raises on bad input.

    Returns a clean dictionary ready for payload building.
    """
    if not isinstance(result, dict):
        logger.warning("validate_detection_result received non-dict: %s", type(result))
        return {
            "elephant_detected": False,
            "adult_count": 0,
            "calf_count": 0,
            "elephant_count": 0,
            "group_type": "herd",
            "behavior": "calm",
            "timestamp": datetime.now().isoformat(),
        }

    # --- elephant_detected ---
    raw_detected = result.get("elephant_detected")
    if isinstance(raw_detected, bool):
        elephant_detected = raw_detected
    elif isinstance(raw_detected, (int, float)):
        elephant_detected = bool(raw_detected)
    elif isinstance(raw_detected, str):
        elephant_detected = raw_detected.strip().lower() in ("true", "1", "yes")
    else:
        elephant_detected = False

    # --- counts ---
    def _safe_int(value, default=0):
        if value is None:
            return default
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return default

    adult_count = _safe_int(result.get("adult_count"), 0)
    calf_count = _safe_int(result.get("calf_count"), 0)
    raw_elephant_count = result.get("elephant_count")

    if raw_elephant_count is not None:
        elephant_count = _safe_int(raw_elephant_count, adult_count + calf_count)
    else:
        elephant_count = adult_count + calf_count

    if elephant_detected and elephant_count == 0:
        elephant_count = 1
        if adult_count == 0 and calf_count == 0:
            adult_count = 1

    # --- behavior ---
    behavior = normalize_behavior_label(result.get("behavior"))

    # --- group_type ---
    raw_group = result.get("group_type", "")
    group_type = normalize_group_type(raw_group, adult_count, calf_count, elephant_count)

    # --- timestamp ---
    timestamp = result.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        timestamp = datetime.now().isoformat()

    return {
        "elephant_detected": elephant_detected,
        "adult_count": adult_count,
        "calf_count": calf_count,
        "elephant_count": elephant_count,
        "group_type": group_type,
        "behavior": behavior,
        "timestamp": timestamp,
    }

# ===========================================================================
# 4. build_payload
# ===========================================================================

def build_payload(result: dict) -> dict:
    """
    Build the compact JSON-ready payload dictionary from a validated result.

    Keys are kept short to minimise bytes over UART.
    """
    return {
        "pillar_id": PILLAR_ID,
        "lat": PILLAR_LATITUDE,
        "lon": PILLAR_LONGITUDE,
        "elephant": result.get("elephant_detected", True),
        "count": result.get("elephant_count", 0),
        "adult": result.get("adult_count", 0),
        "calf": result.get("calf_count", 0),
        "group": result.get("group_type", "herd"),
        "behavior": result.get("behavior", "calm"),
        "ts": result.get("timestamp", datetime.now().isoformat()),
    }

# ===========================================================================
# 5. send_serial_to_esp32
# ===========================================================================

def send_serial_to_esp32(payload: dict) -> bool:
    """
    Serialize *payload* to JSON and send it to the ESP32 over UART.

    The message is terminated with a newline character so the ESP32 can
    detect end-of-message easily.

    Returns True if the write succeeded, False otherwise.
    """
    try:
        import serial as pyserial  # pyserial — imported inside function for clarity
    except ImportError:
        logger.error(
            "pyserial is not installed. Install with: pip install pyserial"
        )
        return False

    json_string = json.dumps(payload, separators=(",", ":"))

    try:
        ser = pyserial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        logger.info("Serial port opened: %s @ %d baud", SERIAL_PORT, BAUD_RATE)

        message = json_string + "\n"
        ser.write(message.encode("utf-8"))
        ser.flush()

        logger.info("Sent to ESP32 (%d bytes): %s", len(message), json_string)

        ser.close()
        logger.info("Serial port closed")
        return True

    except Exception as exc:
        logger.error("Serial send failed: %s", exc)
        return False

# ===========================================================================
# 6. process_and_send_to_esp32  —  MAIN ENTRY POINT
# ===========================================================================

def process_and_send_to_esp32(result: dict) -> dict:
    """
    Main entry point called from the AI pipeline after inference completes.

    Steps:
        1. Validate the result.
        2. Build payload:
           - elephant_detected True  → full payload with all fields.
           - elephant_detected False → minimal payload (pillar_id + elephant=false).
        3. Send via serial to ESP32.
        4. Return structured status dictionary.

    Args:
        result: Raw detection result dictionary from the pipeline.

    Returns:
        {
            "sent": bool,
            "transport": "SERIAL" | None,
            "reason": str,
            "payload": dict | None,
        }
    """
    # Step 1: Validate
    validated = validate_detection_result(result)
    logger.info("Validated result: %s", validated)

    # Step 2: Build payload (full or minimal)
    if not validated["elephant_detected"]:
        payload = {
            "pillar_id": PILLAR_ID,
            "elephant": False,
        }
        logger.info("No elephant detected — sending minimal status payload")
    else:
        payload = build_payload(validated)
    logger.info("Payload built: %s", payload)

    # Step 3: Send via serial
    success = send_serial_to_esp32(payload)

    if success:
        logger.info("Message sent to ESP32 via serial")
        return {
            "sent": True,
            "transport": "SERIAL",
            "reason": "Message sent to ESP32",
            "payload": payload,
        }

    logger.error("Failed to send message to ESP32")
    return {
        "sent": False,
        "transport": "SERIAL",
        "reason": "Serial send failed",
        "payload": payload,
    }


# ===========================================================================
# Self-test / Example usage
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ------------------------------------------------------------------
    # For testing on a machine without /dev/serial0 (e.g. Windows or
    # a Pi without the ESP32 connected), we monkey-patch
    # send_serial_to_esp32 to just log instead of opening the port.
    # ------------------------------------------------------------------
    _original_send = send_serial_to_esp32

    def _mock_serial_send(payload: dict) -> bool:
        """Simulated serial send — prints payload instead of writing UART."""
        json_string = json.dumps(payload, separators=(",", ":"))
        logger.info("[MOCK SERIAL] Would send (%d bytes): %s",
                     len(json_string) + 1, json_string)
        return True

    globals()["send_serial_to_esp32"] = _mock_serial_send

    # ------------------------------------------------------------------
    # Test Case 1: elephant_detected = True → should send
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 1: Elephant detected — message should be sent to ESP32")
    print("=" * 70)

    test_1 = {
        "elephant_detected": True,
        "adult_count": 2,
        "calf_count": 1,
        "elephant_count": 3,
        "group_type": "family",
        "behavior": "calm",
        "timestamp": "2026-03-08T10:30:00",
    }
    status_1 = process_and_send_to_esp32(test_1)
    print("Result:", json.dumps(status_1, indent=2, default=str))
    assert status_1["sent"] is True, "TEST 1 FAILED: expected sent=True"
    assert status_1["transport"] == "SERIAL"
    assert status_1["payload"]["group"] == "family"
    assert status_1["payload"]["behavior"] == "calm"
    print("TEST 1 PASSED\n")

    # ------------------------------------------------------------------
    # Test Case 2: elephant_detected = False → sends minimal payload
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 2: No elephant — sends pillar_id + elephant=false")
    print("=" * 70)

    test_2 = {
        "elephant_detected": False,
        "adult_count": 0,
        "calf_count": 0,
        "elephant_count": 0,
        "group_type": "",
        "behavior": "",
        "timestamp": "2026-03-08T12:00:00",
    }
    status_2 = process_and_send_to_esp32(test_2)
    print("Result:", json.dumps(status_2, indent=2, default=str))
    assert status_2["sent"] is True, "TEST 2 FAILED: expected sent=True"
    assert status_2["transport"] == "SERIAL"
    assert status_2["payload"]["elephant"] is False
    assert status_2["payload"]["pillar_id"] == PILLAR_ID
    assert "count" not in status_2["payload"], "Minimal payload must not have count"
    print("TEST 2 PASSED\n")

    # ------------------------------------------------------------------
    # Test Case 3: behavior "Normal" must be normalised to "calm"
    # ------------------------------------------------------------------
    print("=" * 70)
    print('TEST 3: behavior "Normal" → must become "calm"')
    print("=" * 70)

    test_3 = {
        "elephant_detected": True,
        "adult_count": 3,
        "calf_count": 0,
        "elephant_count": 3,
        "group_type": "herd",
        "behavior": "Normal",
        "timestamp": "2026-03-08T14:00:00",
    }
    status_3 = process_and_send_to_esp32(test_3)
    print("Result:", json.dumps(status_3, indent=2, default=str))
    assert status_3["sent"] is True
    assert status_3["payload"]["behavior"] == "calm", \
        f"TEST 3 FAILED: behavior={status_3['payload']['behavior']}"
    print("TEST 3 PASSED\n")

    # ------------------------------------------------------------------
    # Test Case 4: group_type correction — 1 elephant → "individual"
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 4: group_type correction — count=1 → individual")
    print("=" * 70)

    test_4 = {
        "elephant_detected": True,
        "adult_count": 1,
        "calf_count": 0,
        "elephant_count": 1,
        "group_type": "herd",      # wrong — should become "individual"
        "behavior": "aggressive",
        "timestamp": "2026-03-08T15:00:00",
    }
    status_4 = process_and_send_to_esp32(test_4)
    print("Result:", json.dumps(status_4, indent=2, default=str))
    assert status_4["payload"]["group"] == "individual", \
        f"TEST 4 FAILED: group={status_4['payload']['group']}"
    print("TEST 4 PASSED\n")

    # ------------------------------------------------------------------
    # Test Case 5: empty dict — must not crash
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 5: Empty dict — must not crash, elephant_detected defaults False")
    print("=" * 70)

    status_5 = process_and_send_to_esp32({})
    print("Result:", json.dumps(status_5, indent=2, default=str))
    assert status_5["sent"] is True, "TEST 5 FAILED: expected sent=True (minimal payload)"
    assert status_5["payload"]["elephant"] is False
    print("TEST 5 PASSED\n")

    # ------------------------------------------------------------------
    # Test Case 6: elephant True with missing fields — safe defaults
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 6: elephant=True but all other fields missing")
    print("=" * 70)

    status_6 = process_and_send_to_esp32({"elephant_detected": True})
    print("Result:", json.dumps(status_6, indent=2, default=str))
    assert status_6["sent"] is True
    assert status_6["payload"]["adult"] >= 1, "Should assume at least 1 adult"
    assert status_6["payload"]["behavior"] == "calm"
    print("TEST 6 PASSED\n")

    # Restore original
    globals()["send_serial_to_esp32"] = _original_send

    print("=" * 70)
    print("All 6 tests passed.")
    print("=" * 70)
