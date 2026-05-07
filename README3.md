#include <SPI.h>
#include <LoRa.h>
#include <ArduinoJson.h>

// ================================================================
// ESP32 LoRa SENDER — Elephant Detection System
// ------------------------------------------------
// Raspberry Pi sends one JSON object per line over USB serial.
// This ESP32 converts it to the compact receiver format and sends
// it over LoRa with a sequence number and ACK retry handling.
// ================================================================

// ---------------- LoRa pins ----------------
#define LORA_SS    5
#define LORA_RST   14
#define LORA_DIO0  26

// ---------------- LoRa radio settings ----------------
#define LORA_FREQUENCY     433E6
#define LORA_BANDWIDTH     125E3
#define LORA_SPREAD_FACTOR 7
#define LORA_CODING_RATE   5
#define LORA_TX_POWER      20
#define LORA_SYNC_WORD     0x12

// ---------------- Optional status LED ----------------
#ifndef LED_BUILTIN
#define LED_BUILTIN 2
#endif

// ---------------- Serial / retry settings ----------------
static const unsigned long SERIAL_BAUD_RATE = 115200;
static const unsigned long ACK_TIMEOUT_MS   = 2000;
static const int MAX_RETRIES = 3;

// ---------------- Default pillar identity ----------------
// Update these to match the physical pillar location.
static const char* DEFAULT_PILLAR_ID = "PILLAR_02";
static const double DEFAULT_PILLAR_LAT = 7.123456;
static const double DEFAULT_PILLAR_LON = 80.123456;

// ---------------- State ----------------
uint16_t sequenceNumber = 1;

struct DetectionPayload {
  bool elephantDetected = false;
  int adultCount = 0;
  int calfCount = 0;
  int elephantCount = 0;
  String groupType = "herd";
  String behavior = "calm";
  String timestamp = "";
  String pillarId = DEFAULT_PILLAR_ID;
  double lat = DEFAULT_PILLAR_LAT;
  double lon = DEFAULT_PILLAR_LON;
};

static String readLineFromSerial() {
  if (!Serial.available()) {
    return String();
  }

  String line = Serial.readStringUntil('\n');
  line.trim();
  return line;
}

static String jsonStringOr(const JsonDocument& doc, const char* key, const String& fallback) {
  if (doc.containsKey(key)) {
    const char* value = doc[key];
    if (value && value[0] != '\0') {
      return String(value);
    }
  }
  return fallback;
}

static int jsonIntOr(const JsonDocument& doc, const char* key, int fallback) {
  if (doc.containsKey(key)) {
    return doc[key].as<int>();
  }
  return fallback;
}

static double jsonDoubleOr(const JsonDocument& doc, const char* key, double fallback) {
  if (doc.containsKey(key)) {
    return doc[key].as<double>();
  }
  return fallback;
}

static bool jsonBoolOr(const JsonDocument& doc, const char* key, bool fallback) {
  if (doc.containsKey(key)) {
    return doc[key].as<bool>();
  }
  return fallback;
}

static bool isValidGps(double lat, double lon) {
  if (lat == 0.0 && lon == 0.0) return false;
  if (lat < -90.0 || lat > 90.0) return false;
  if (lon < -180.0 || lon > 180.0) return false;
  if (isnan(lat) || isnan(lon)) return false;
  if (isinf(lat) || isinf(lon)) return false;
  return true;
}

static String normalizeBehavior(String value) {
  value.trim();
  value.toLowerCase();

  if (value == "aggressive" || value == "agitated" || value == "threat" ||
      value == "threatening" || value == "trumpet" || value == "trumpeting" ||
      value == "charging" || value == "attack") {
    return "aggressive";
  }

  return "calm";
}

static String normalizeGroupType(String groupType, int adultCount, int calfCount, int elephantCount) {
  groupType.trim();
  groupType.toLowerCase();

  adultCount = max(adultCount, 0);
  calfCount = max(calfCount, 0);
  elephantCount = max(elephantCount, 0);

  if (elephantCount == 1) {
    return "individual";
  }

  if (adultCount == 2 && calfCount >= 1) {
    return "family";
  }

  if (elephantCount > 1) {
    return "herd";
  }

  if (groupType == "individual" || groupType == "family" || groupType == "herd") {
    return groupType;
  }

  return "herd";
}

static bool parseInputJson(const String& input, DetectionPayload& payload) {
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, input);
  if (err) {
    Serial.printf("[PARSE] ✗ Invalid JSON: %s\n", err.c_str());
    Serial.printf("[PARSE] Input was: %s\n", input.c_str());
    Serial.flush();
    return false;
  }
  
  Serial.println("[PARSE] ✓ JSON parsed successfully!");

  payload.pillarId = jsonStringOr(doc, "pillar_id", DEFAULT_PILLAR_ID);

  double inLat = jsonDoubleOr(doc, "lat", DEFAULT_PILLAR_LAT);
  double inLon = jsonDoubleOr(doc, "lon", DEFAULT_PILLAR_LON);
  if (isValidGps(inLat, inLon)) {
    payload.lat = inLat;
    payload.lon = inLon;
  }

  payload.elephantDetected = jsonBoolOr(doc, "elephant_detected", jsonBoolOr(doc, "elephant", false));
  payload.adultCount = jsonIntOr(doc, "adult_count", jsonIntOr(doc, "adult", 0));
  payload.calfCount = jsonIntOr(doc, "calf_count", jsonIntOr(doc, "calf", 0));
  payload.elephantCount = jsonIntOr(doc, "elephant_count", jsonIntOr(doc, "count", payload.adultCount + payload.calfCount));

  String behavior = jsonStringOr(doc, "behavior", "calm");
  String groupType = jsonStringOr(doc, "group_type", jsonStringOr(doc, "group", "herd"));
  payload.behavior = normalizeBehavior(behavior);
  payload.groupType = normalizeGroupType(groupType, payload.adultCount, payload.calfCount, payload.elephantCount);

  payload.timestamp = jsonStringOr(doc, "timestamp", jsonStringOr(doc, "ts", ""));
  if (payload.timestamp.length() == 0) {
    payload.timestamp = String(millis());
  }

  if (payload.elephantDetected && payload.elephantCount <= 0) {
    payload.elephantCount = max(1, payload.adultCount + payload.calfCount);
  }

  return true;
}

static String buildReceiverJson(const DetectionPayload& payload) {
  StaticJsonDocument<256> doc;
  doc["pillar_id"] = payload.pillarId;
  doc["lat"] = payload.lat;
  doc["lon"] = payload.lon;
  doc["elephant"] = payload.elephantDetected;
  doc["count"] = payload.elephantCount;
  doc["adult"] = payload.adultCount;
  doc["calf"] = payload.calfCount;
  doc["group"] = payload.groupType;
  doc["behavior"] = payload.behavior;
  doc["ts"] = payload.timestamp;

  String output;
  serializeJson(doc, output);
  return output;
}

static String buildSequencedFrame(uint16_t seq, const String& jsonPayload) {
  return String("SEQ:") + String(seq) + "|" + jsonPayload;
}

static void blinkLed(int times, int onMs, int offMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(onMs);
    digitalWrite(LED_BUILTIN, LOW);
    delay(offMs);
  }
}

static bool waitForAck(uint16_t seq, unsigned long timeoutMs) {
  unsigned long start = millis();
  String expected = String("ACK:") + String(seq);

  while ((millis() - start) < timeoutMs) {
    int packetSize = LoRa.parsePacket();
    if (packetSize <= 0) {
      delay(10);
      continue;
    }

    String response;
    while (LoRa.available()) {
      response += (char)LoRa.read();
    }
    response.trim();

    if (response == expected) {
      Serial.printf("[LORA] ACK received for seq=%u\n", seq);
      LoRa.receive();
      return true;
    }
  }

  LoRa.receive();
  return false;
}

static bool sendLoRaFrame(const String& frame, uint16_t seq) {
  for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    Serial.printf("[LORA] TX attempt %d/%d seq=%u\n", attempt, MAX_RETRIES, seq);

    LoRa.idle();
    LoRa.beginPacket();
    LoRa.print(frame);
    LoRa.endPacket(true);
    LoRa.receive();

    if (waitForAck(seq, ACK_TIMEOUT_MS)) {
      blinkLed(1, 40, 20);
      return true;
    }

    Serial.println("[LORA] ACK timeout, retrying...");
    blinkLed(2, 60, 60);
  }

  Serial.printf("[LORA] Failed after %d attempts\n", MAX_RETRIES);
  return false;
}

static void handleSerialMessage(const String& message) {
  if (message.length() == 0) {
    return;
  }

  DetectionPayload payload;
  if (!parseInputJson(message, payload)) {
    return;
  }

  String receiverJson = buildReceiverJson(payload);
  String frame = buildSequencedFrame(sequenceNumber, receiverJson);

  Serial.printf("[SERIAL] Parsed pillar=%s elephant=%s group=%s behavior=%s\n",
                payload.pillarId.c_str(),
                payload.elephantDetected ? "true" : "false",
                payload.groupType.c_str(),
                payload.behavior.c_str());
  Serial.printf("[SERIAL] LoRa frame: %s\n", frame.c_str());

  if (sendLoRaFrame(frame, sequenceNumber)) {
    Serial.printf("[OK] Sent seq=%u\n", sequenceNumber);
    sequenceNumber++;
  } else {
    Serial.printf("[ERROR] Transmission failed for seq=%u\n", sequenceNumber);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Initialize Serial with longer delay to allow bootloader cleanup
  Serial.begin(SERIAL_BAUD_RATE);
  delay(500);
  
  // Force a fresh line and clear any bootloader garbage
  Serial.println();
  Serial.println();
  delay(100);
  
  // Unmistakable startup banner
  Serial.println();
  Serial.println("====================================================");
  Serial.println("  *** ESP32 LoRa SENDER STARTING ***");
  Serial.println("  Elephant Detection — Raspberry Pi -> LoRa");
  Serial.println("====================================================");
  Serial.println();

  SPI.begin(18, 19, 23, LORA_SS);
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);

  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("[ERROR] LoRa init failed. Check wiring.");
    while (true) {
      blinkLed(1, 150, 150);
    }
  }

  LoRa.setSpreadingFactor(LORA_SPREAD_FACTOR);
  LoRa.setSignalBandwidth(LORA_BANDWIDTH);
  LoRa.setCodingRate4(LORA_CODING_RATE);
  LoRa.setTxPower(LORA_TX_POWER);
  LoRa.setSyncWord(LORA_SYNC_WORD);
  LoRa.enableCrc();
  LoRa.receive();

  Serial.println("[OK] LoRa radio online and listening.");
  Serial.print("    Frequency: "); Serial.print(LORA_FREQUENCY / 1E6, 1); Serial.println(" MHz");
  Serial.print("    Spread Factor: "); Serial.println(LORA_SPREAD_FACTOR);
  Serial.print("    Bandwidth: "); Serial.print(LORA_BANDWIDTH / 1E3, 0); Serial.println(" kHz");
  Serial.print("    Sync Word: 0x"); Serial.println(LORA_SYNC_WORD, HEX);
  Serial.println();
  Serial.println("Waiting for detection JSON from Raspberry Pi...");
  Serial.println("(Send one complete JSON object per line, ending with newline)");
  Serial.println();
  Serial.println("Test example - paste this in a serial terminal:");
  Serial.println("{\"elephant_detected\":true,\"adult_count\":2,\"calf_count\":1,\"group_type\":\"family\",\"behavior\":\"calm\"}");
  Serial.println();
  Serial.println("[TROUBLESHOOTING]");
  Serial.println("  If heartbeat repeats but no JSON arrives:");
  Serial.println("  1. Check baud rate = 115200");
  Serial.println("  2. Verify USB cable is connected");
  Serial.println("  3. Check Raspberry Pi is sending to correct port");
  Serial.println("----------------------------------------------------");
  Serial.flush();
  
  // Three quick blinks = ready
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(50);
    digitalWrite(LED_BUILTIN, LOW);
    delay(50);
  }
}

void loop() {
  // Check for any incoming data (debug)
  if (Serial.available()) {
    int bytes_available = Serial.available();
    Serial.printf("[DEBUG] %d bytes available on serial\n", bytes_available);
  }
  
  String line = readLineFromSerial();
  if (line.length() > 0) {
    Serial.printf("[DEBUG] Raw line received (%d chars): ", line.length());
    if (line.length() > 100) {
      Serial.print(line.substring(0, 100));
      Serial.println("...");
    } else {
      Serial.println(line);
    }
    Serial.flush();
    
    handleSerialMessage(line);
    Serial.flush();
  } else {
    // Tiny heartbeat every 5 seconds so user knows sketch is alive
    static unsigned long lastHeartbeat = 0;
    if ((millis() - lastHeartbeat) > 5000) {
      lastHeartbeat = millis();
      Serial.println("[HEARTBEAT] Waiting for serial input from Raspberry Pi...");
      Serial.flush();
    }
  }
}