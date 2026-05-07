#include <SPI.h>
#include <LoRa.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <math.h>

// ── LoRa pin definitions (must match sender) ────────────────────────────────
#define LORA_SS    5
#define LORA_RST   14
#define LORA_DIO0  26

// ── LoRa radio settings (MUST match sender exactly) ────────────────────────
#define LORA_FREQUENCY      433E6
#define LORA_BANDWIDTH      125E3
#define LORA_SPREAD_FACTOR  7
#define LORA_CODING_RATE    5
#define LORA_TX_POWER       20
#define LORA_SYNC_WORD      0x12

// ── Output pins ─────────────────────────────────────────────────────────────
#define LED_PIN     2   // Built-in LED
#define BUZZER_PIN  25  // Buzzer for aggressive behavior alert

// ── WiFi credentials ────────────────────────────────────────────────────────
const char* ssid     = "Mali";
const char* password = "1812@pjtm";

WebServer server(80);

// ── Tracking ────────────────────────────────────────────────────────────────
unsigned long totalReceived    = 0;
unsigned long elephantAlerts   = 0;
unsigned long aggressiveAlerts = 0;
uint16_t lastSeq = 0;

// ── Latest detection state ──────────────────────────────────────────────────
struct DetectionState {
  bool    elephantDetected;
  int     count;
  int     adult;
  int     calf;
  char    pillarId[20];
  char    group[16];
  char    behavior[16];
  float   lat;
  float   lon;
  char    timestamp[30];
  int     rssi;
  float   snr;
} latestDetection;

// ── Pillar structure (populated from LoRa alerts) ───────────────────────────
struct Pillar {
  String id;
  String name;
  double lat;
  double lon;
  bool   active;
  bool   elephantDetected;
  String detectedAt;
  int    elephantCount;
  int    elephantCalf;
  int    elephantAdult;
  String elephantBehavior;
  String elephantGroup;
};

const int MAX_PILLARS = 50;
Pillar pillars[MAX_PILLARS];
int pillarCount = 0;

// ── Phone GPS location (received from mobile app) ──────────────────────────
double phoneLat = 0.0;
double phoneLon = 0.0;
double phoneGPSAccuracy = 0.0;       // accuracy in metres from phone
bool   phoneGPSReceived = false;
unsigned long lastGPSUpdateTime  = 0; // millis() when last GPS received
unsigned long lastStatusPrintTime = 0;
const unsigned long STATUS_PRINT_INTERVAL = 30000; // 30s

// ── Reliability constants ───────────────────────────────────────────────────
const unsigned long PHONE_GPS_STALE_MS  = 30000;  // phone GPS stale after 30s
const unsigned long PILLAR_STALE_MS     = 300000;  // pillar data stale after 5 min
const unsigned long WIFI_RECONNECT_MS   = 15000;  // WiFi reconnect interval
const float GPS_ACCURACY_REJECT_M       = 100.0;  // reject phone GPS with accuracy > 100m
unsigned long lastWiFiReconnectAttempt  = 0;
unsigned long loraLastReceivedTime      = 0;       // millis() when last LoRa packet received

// ── Per-pillar freshness tracking ───────────────────────────────────────────
unsigned long pillarLastUpdate[MAX_PILLARS]; // millis() timestamp per pillar

// ── Cached distance/risk ────────────────────────────────────────────────────
double cachedNearestDistance  = -1;    // metres, -1 = not yet calculated
String cachedRiskLevel        = "none";
String cachedNearestPillarId  = "";

// =========================================================================
// GPS Coordinate Validation
// Returns true only if lat/lon are within valid Earth ranges and non-zero
// =========================================================================
bool isValidGPS(double lat, double lon) {
  if (lat == 0.0 && lon == 0.0) return false;         // zero = no fix
  if (lat < -90.0 || lat > 90.0) return false;        // latitude range
  if (lon < -180.0 || lon > 180.0) return false;      // longitude range
  if (isnan(lat) || isnan(lon)) return false;          // NaN check
  if (isinf(lat) || isinf(lon)) return false;          // infinity check
  return true;
}

// =========================================================================
// Check if phone GPS data is still fresh (not stale)
// =========================================================================
bool isPhoneGPSFresh() {
  if (!phoneGPSReceived) return false;
  return (millis() - lastGPSUpdateTime) < PHONE_GPS_STALE_MS;
}

// =========================================================================
// Check if a pillar's data is still fresh
// =========================================================================
bool isPillarFresh(int idx) {
  if (idx < 0 || idx >= pillarCount) return false;
  return (millis() - pillarLastUpdate[idx]) < PILLAR_STALE_MS;
}

// =========================================================================
// Mark stale pillars as inactive (called periodically)
// =========================================================================
void expireStalePillars() {
  for (int i = 0; i < pillarCount; i++) {
    if (pillars[i].active && pillars[i].elephantDetected) {
      if ((millis() - pillarLastUpdate[i]) > PILLAR_STALE_MS) {
        pillars[i].elephantDetected = false;
        Serial.printf("  [STALE] Pillar %s marked inactive (no update for %lu s)\n",
                      pillars[i].id.c_str(), PILLAR_STALE_MS / 1000);
      }
    }
  }
}

// =========================================================================
// Haversine Distance Calculation
// Calculates the great-circle distance between two GPS coordinates
// Uses: phone GPS (lat1, lon1) and pillar GPS (lat2, lon2)
// Returns: distance in metres
// =========================================================================
double haversineDistance(double lat1, double lon1, double lat2, double lon2) {
  const double R = 6371000.0; // Earth radius in metres
  double dLat = (lat2 - lat1) * PI / 180.0;
  double dLon = (lon2 - lon1) * PI / 180.0;

  double radLat1 = lat1 * PI / 180.0;
  double radLat2 = lat2 * PI / 180.0;

  double a = sin(dLat / 2.0) * sin(dLat / 2.0) +
             cos(radLat1) * cos(radLat2) *
             sin(dLon / 2.0) * sin(dLon / 2.0);
  double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));

  return R * c; // distance in metres
}

// =========================================================================
// Find nearest pillar to given GPS coordinates
// =========================================================================
int findNearestPillar(double lat, double lon) {
  if (pillarCount == 0) return -1;

  int    nearestIndex = -1;
  double minDistance   = 999999999.0;

  for (int i = 0; i < pillarCount; i++) {
    if (!pillars[i].active) continue;
    double dist = haversineDistance(lat, lon, pillars[i].lat, pillars[i].lon);
    if (dist < minDistance) {
      minDistance   = dist;
      nearestIndex = i;
    }
  }
  return nearestIndex;
}

// =========================================================================
// Risk Level Calculation
//   < 2 km  → high (always)
//   2–6 km  → medium (but high if aggressive or family)
//   6–10 km → low
//   > 10 km → none
// =========================================================================
String calculateRiskLevel(double distanceKm, const String& behavior, const String& group) {
  if (distanceKm < 2.0) {
    return "high";
  }
  if (distanceKm <= 6.0) {
    if (behavior == "aggressive" || group == "family") {
      return "high";
    }
    return "medium";
  }
  if (distanceKm <= 10.0) {
    return "low";
  }
  return "none";
}

// =========================================================================
// Recalculate distance & risk for all active elephant pillars
// Called after every new phone GPS or LoRa event
// Uses Haversine: phone GPS ↔ each pillar GPS
// =========================================================================
void recalculateDistances() {
  if (!isPhoneGPSFresh() || pillarCount == 0) return;
  if (!isValidGPS(phoneLat, phoneLon)) return;

  double minDist = 999999999.0;
  int    nearIdx = -1;

  for (int i = 0; i < pillarCount; i++) {
    if (!pillars[i].elephantDetected || !pillars[i].active) continue;
    if (!isPillarFresh(i)) continue;  // skip stale pillar data
    if (!isValidGPS(pillars[i].lat, pillars[i].lon)) continue;

    // Haversine: phone GPS → pillar GPS
    double d = haversineDistance(phoneLat, phoneLon, pillars[i].lat, pillars[i].lon);
    if (d < minDist) {
      minDist = d;
      nearIdx = i;
    }
  }

  if (nearIdx >= 0) {
    cachedNearestDistance  = minDist;
    cachedNearestPillarId = pillars[nearIdx].id;
    cachedRiskLevel = calculateRiskLevel(
      minDist / 1000.0,
      pillars[nearIdx].elephantBehavior,
      pillars[nearIdx].elephantGroup
    );
  } else {
    cachedNearestDistance  = -1;
    cachedRiskLevel        = "none";
    cachedNearestPillarId  = "";
  }
}

// =========================================================================
// Store LoRa alert into pillar memory
// =========================================================================
void storePillarAlert(const char* jsonStr) {
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, jsonStr);
  if (err) {
    Serial.printf("  [ERROR] storePillarAlert JSON parse failed: %s\n", err.c_str());
    return;
  }

  const char* pillar_id = doc["pillar_id"] | "";
  float pillarLat  = doc["lat"]      | 0.0;
  float pillarLon  = doc["lon"]      | 0.0;
  bool  elephant   = doc["elephant"] | false;
  int   count      = doc["count"]    | 0;
  int   adult      = doc["adult"]    | 0;
  int   calf       = doc["calf"]     | 0;
  const char* group    = doc["group"]    | "";
  const char* behavior = doc["behavior"] | "";
  const char* ts       = doc["ts"]       | "";

  // Validate sender GPS coordinates — reject invalid/zero coordinates
  if (!isValidGPS(pillarLat, pillarLon)) {
    Serial.printf("  [REJECT] Invalid sender GPS: %.6f, %.6f — skipping pillar store\n",
                  pillarLat, pillarLon);
    return;
  }

  // Validate pillar_id is not empty
  if (strlen(pillar_id) == 0) {
    Serial.println("  [WARN] Empty pillar_id in LoRa data — using lat/lon as key");
  }

  // Find matching pillar by pillar_id
  int targetIdx = -1;
  for (int i = 0; i < pillarCount; i++) {
    if (pillars[i].id == String(pillar_id)) {
      targetIdx = i;
      break;
    }
  }

  // If not found by ID, find nearest by lat/lon
  if (targetIdx < 0 && pillarLat != 0.0 && pillarLon != 0.0) {
    double minDist = 999999999.0;
    for (int i = 0; i < pillarCount; i++) {
      double dist = haversineDistance(pillarLat, pillarLon, pillars[i].lat, pillars[i].lon);
      if (dist < minDist) {
        minDist   = dist;
        targetIdx = i;
      }
    }
  }

  // Create new pillar if not found
  if (targetIdx < 0 && pillarCount < MAX_PILLARS) {
    targetIdx = pillarCount;
    pillars[targetIdx].id     = String(pillar_id);
    pillars[targetIdx].name   = String(pillar_id);
    pillars[targetIdx].lat    = pillarLat;
    pillars[targetIdx].lon    = pillarLon;
    pillars[targetIdx].active = true;
    pillarCount++;
  }

  if (targetIdx < 0) return;

  pillars[targetIdx].elephantDetected = elephant;
  pillars[targetIdx].elephantCount    = count;
  pillars[targetIdx].elephantAdult    = adult;
  pillars[targetIdx].elephantCalf     = calf;
  pillars[targetIdx].elephantBehavior = String(behavior);
  pillars[targetIdx].elephantGroup    = String(group);
  pillars[targetIdx].detectedAt       = String(ts);
  pillarLastUpdate[targetIdx]         = millis();  // freshness timestamp

  // Always update GPS from sender (already validated above)
  pillars[targetIdx].lat = pillarLat;
  pillars[targetIdx].lon = pillarLon;

  Serial.printf("  [STORE] Pillar %s updated — elephant=%s, GPS=(%.6f, %.6f), count=%d\n",
                pillars[targetIdx].id.c_str(),
                elephant ? "YES" : "NO",
                pillarLat, pillarLon, count);

  // Auto-recalculate distances when a new LoRa alert arrives
  if (isPhoneGPSFresh()) {
    recalculateDistances();
    Serial.printf("  [AUTO] Haversine recalculated: %.2f m (%s) to %s\n",
                  cachedNearestDistance,
                  cachedRiskLevel.c_str(),
                  cachedNearestPillarId.c_str());
  }
}

// =========================================================================
// API Handlers (for mobile app)
// =========================================================================

// ── GET / — Device info ─────────────────────────────────────────────────
void handleRoot() {
  String ipAddress = WiFi.localIP().toString();
  if (WiFi.getMode() == WIFI_AP) {
    ipAddress = WiFi.softAPIP().toString();
  }

  DynamicJsonDocument response(512);
  response["status"]  = "ready";
  response["system"]  = "ESP32 Elephant Detection";
  response["ip"]      = ipAddress;
  response["pillars"] = pillarCount;
  response["message"] = "Connect your mobile app to: " + ipAddress;

  String output;
  serializeJson(response, output);
  server.send(200, "application/json", output);
}

// =========================================================================
// POST /gps — Receive phone GPS, calculate Haversine distance, respond
//
// Mobile app sends:  { "lat": <phone_lat>, "lon": <phone_lon>, "accuracy": <m> }
//
// ESP32 calculates:  haversineDistance(phone_lat, phone_lon, pillar_lat, pillar_lon)
//
// ESP32 responds with:
//   - distance (metres & km) between phone and nearest elephant pillar
//   - pillar info (id, name, lat, lon, elephant count, behavior, group)
//   - risk level based on distance + behavior + group
//   - all active elephant alerts with individual distances
// =========================================================================
void handleGPS() {
  if (!server.hasArg("plain")) {
    server.send(400, "application/json", "{\"error\":\"No data\"}");
    return;
  }

  String body = server.arg("plain");
  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, body);

  if (error) {
    server.send(400, "application/json", "{\"error\":\"Invalid JSON\"}");
    return;
  }

  // ── Validate phone GPS coordinates ──────────────────────────────────
  double inLat = doc["lat"] | 0.0;
  double inLon = doc["lon"] | 0.0;
  double inAcc = doc["accuracy"] | 0.0;

  if (!isValidGPS(inLat, inLon)) {
    Serial.printf("  [REJECT] Invalid phone GPS: %.8f, %.8f\n", inLat, inLon);
    server.send(400, "application/json",
      "{\"error\":\"Invalid GPS coordinates\",\"detail\":\"lat must be -90..90, lon -180..180, non-zero\"}");
    return;
  }

  // Reject phone GPS with very poor accuracy
  if (inAcc > GPS_ACCURACY_REJECT_M && inAcc > 0) {
    Serial.printf("  [REJECT] Phone GPS accuracy too poor: %.1f m (limit: %.1f m)\n",
                  inAcc, GPS_ACCURACY_REJECT_M);
    server.send(400, "application/json",
      "{\"error\":\"GPS accuracy too poor\",\"accuracy_m\":" + String(inAcc, 1) +
      ",\"limit_m\":" + String(GPS_ACCURACY_REJECT_M, 1) + "}");
    return;
  }

  // ── Store phone GPS coordinates ───────────────────────────────────────
  phoneLat         = inLat;
  phoneLon         = inLon;
  phoneGPSAccuracy = inAcc;
  phoneGPSReceived = true;
  lastGPSUpdateTime = millis();

  // ── Expire stale pillar data before calculation ───────────────────────
  expireStalePillars();

  // ── Recalculate all distances using Haversine (phone GPS ↔ pillar GPS)
  recalculateDistances();

  // ── Find nearest elephant pillar using Haversine ──────────────────────
  int    elephantPillarIdx = -1;
  double minDistance = 999999999.0;

  for (int i = 0; i < pillarCount; i++) {
    if (pillars[i].elephantDetected && pillars[i].active) {
      if (!isPillarFresh(i)) continue;  // skip stale sender data
      if (!isValidGPS(pillars[i].lat, pillars[i].lon)) continue;  // skip invalid GPS
      // Haversine: phone GPS → pillar GPS
      double dist = haversineDistance(phoneLat, phoneLon, pillars[i].lat, pillars[i].lon);
      if (dist < minDistance) {
        minDistance = dist;
        elephantPillarIdx = i;
      }
    }
  }

  DynamicJsonDocument response(4096);
  response["status"] = "success";

  // ── Include phone GPS in response so app can verify ───────────────────
  response["phoneGPS"]["lat"]      = phoneLat;
  response["phoneGPS"]["lon"]      = phoneLon;
  response["phoneGPS"]["accuracy"] = phoneGPSAccuracy;
  response["phoneGPS"]["fresh"]    = isPhoneGPSFresh();
  response["phoneGPS"]["age_sec"]  = (millis() - lastGPSUpdateTime) / 1000;

  if (elephantPillarIdx < 0) {
    response["elephantDetected"] = false;
    response["message"] = "No elephant detected";

    String output;
    serializeJson(response, output);
    server.send(200, "application/json", output);
    return;
  }

  // ── Elephant found — calculate distance using Haversine ───────────────
  Pillar* ep = &pillars[elephantPillarIdx];

  // Haversine distance: phone GPS → elephant pillar GPS
  double phoneToPillarDistance = haversineDistance(phoneLat, phoneLon, ep->lat, ep->lon);

  // Find nearest pillar to phone (any pillar, not just elephant ones)
  int    nearestPillarIdx      = findNearestPillar(phoneLat, phoneLon);
  double nearestPillarDistance = 0;
  String nearestPillarName     = "None";

  if (nearestPillarIdx >= 0) {
    nearestPillarName     = pillars[nearestPillarIdx].name;
    nearestPillarDistance = haversineDistance(phoneLat, phoneLon,
                                              pillars[nearestPillarIdx].lat,
                                              pillars[nearestPillarIdx].lon);
  }

  // ── Serial log: Haversine calculation details ─────────────────────────
  Serial.println();
  Serial.println("╔══════════════════════════════════════════════╗");
  Serial.println("║       HAVERSINE DISTANCE CALCULATION        ║");
  Serial.println("╠══════════════════════════════════════════════╣");
  Serial.printf("║ Phone GPS  : %.8f, %.8f       ║\n", phoneLat, phoneLon);
  Serial.printf("║ Phone Acc  : %.1f m                          ║\n", phoneGPSAccuracy);
  Serial.printf("║ Pillar GPS : %.8f, %.8f       ║\n", ep->lat, ep->lon);
  Serial.printf("║ Pillar     : %-30s ║\n", ep->name.c_str());
  Serial.printf("║ Distance   : %.2f m (%.2f km)               ║\n",
                phoneToPillarDistance, phoneToPillarDistance / 1000.0);
  Serial.println("╚══════════════════════════════════════════════╝");

  // ── Build JSON response for mobile app ────────────────────────────────
  response["elephantDetected"] = true;

  // Elephant pillar info
  response["elephantLocation"]["lat"]       = ep->lat;
  response["elephantLocation"]["lon"]       = ep->lon;
  response["elephantLocation"]["pillarId"]  = ep->id;
  response["elephantLocation"]["pillarName"]= ep->name;
  response["elephantLocation"]["detectedAt"]= ep->detectedAt;
  response["elephantLocation"]["count"]     = ep->elephantCount;
  response["elephantLocation"]["adult"]     = ep->elephantAdult;
  response["elephantLocation"]["calf"]      = ep->elephantCalf;
  response["elephantLocation"]["behavior"]  = ep->elephantBehavior;
  response["elephantLocation"]["group"]     = ep->elephantGroup;

  // Distance (Haversine: phone GPS ↔ pillar GPS)
  response["distance"]["haversine_m"]       = phoneToPillarDistance;
  response["distance"]["haversine_km"]      = phoneToPillarDistance / 1000.0;
  response["distance"]["track"]             = phoneToPillarDistance;
  response["distance"]["track_km"]          = phoneToPillarDistance / 1000.0;
  response["distance"]["straight"]          = phoneToPillarDistance;
  response["distance"]["straight_km"]       = phoneToPillarDistance / 1000.0;
  response["distance"]["nearestPillar"]     = nearestPillarDistance;
  response["distance"]["nearestPillar_km"]  = nearestPillarDistance / 1000.0;
  response["distance"]["nearestPillarName"] = nearestPillarName;

  // Risk level (based on Haversine distance + elephant behavior + group)
  String riskLevel = calculateRiskLevel(
    phoneToPillarDistance / 1000.0,
    ep->elephantBehavior,
    ep->elephantGroup
  );
  response["riskLevel"] = riskLevel;

  // Update cached values
  cachedNearestDistance  = phoneToPillarDistance;
  cachedRiskLevel        = riskLevel;
  cachedNearestPillarId  = ep->id;

  // ── Include ALL active elephant alerts with individual Haversine distances
  JsonArray allAlerts = response.createNestedArray("allAlerts");
  for (int i = 0; i < pillarCount; i++) {
    if (pillars[i].elephantDetected && pillars[i].active) {
      if (!isValidGPS(pillars[i].lat, pillars[i].lon)) continue;  // skip invalid
      JsonObject a = allAlerts.createNestedObject();
      // Haversine: phone GPS → this pillar GPS
      double d = haversineDistance(phoneLat, phoneLon, pillars[i].lat, pillars[i].lon);
      a["pillarId"]    = pillars[i].id;
      a["pillarName"]  = pillars[i].name;
      a["lat"]         = pillars[i].lat;
      a["lon"]         = pillars[i].lon;
      a["count"]       = pillars[i].elephantCount;
      a["adult"]       = pillars[i].elephantAdult;
      a["calf"]        = pillars[i].elephantCalf;
      a["behavior"]    = pillars[i].elephantBehavior;
      a["group"]       = pillars[i].elephantGroup;
      a["detectedAt"]  = pillars[i].detectedAt;
      a["distance_m"]  = d;
      a["distance_km"] = d / 1000.0;
      a["riskLevel"]   = calculateRiskLevel(d / 1000.0, pillars[i].elephantBehavior, pillars[i].elephantGroup);
      a["fresh"]       = isPillarFresh(i);
      a["age_sec"]     = (millis() - pillarLastUpdate[i]) / 1000;
    }
  }

  String output;
  serializeJson(response, output);
  server.send(200, "application/json", output);
}

// ── GET /status — System status ─────────────────────────────────────────
void handleStatus() {
  DynamicJsonDocument response(2048);
  response["status"]         = "online";
  response["phoneConnected"] = phoneGPSReceived;
  response["pillarCount"]    = pillarCount;
  response["wifiRSSI"]       = WiFi.RSSI();
  response["loraStats"]["totalReceived"]    = totalReceived;
  response["loraStats"]["elephantAlerts"]   = elephantAlerts;
  response["loraStats"]["aggressiveAlerts"] = aggressiveAlerts;
  if (loraLastReceivedTime > 0) {
    response["loraStats"]["lastReceived_sec"] = (millis() - loraLastReceivedTime) / 1000;
  }

  // Cached nearest elephant distance (Haversine)
  if (cachedNearestDistance >= 0) {
    response["nearestElephant"]["distance_m"]  = cachedNearestDistance;
    response["nearestElephant"]["distance_km"] = cachedNearestDistance / 1000.0;
    response["nearestElephant"]["pillarId"]    = cachedNearestPillarId;
    response["nearestElephant"]["riskLevel"]   = cachedRiskLevel;
  }

  // Phone GPS info
  if (phoneGPSReceived) {
    response["phoneLat"]        = phoneLat;
    response["phoneLon"]        = phoneLon;
    response["gpsAccuracy"]     = phoneGPSAccuracy;
    response["gpsFreshnessSec"] = (millis() - lastGPSUpdateTime) / 1000;
    response["gpsFresh"]        = isPhoneGPSFresh();
  }

  // Active elephant alerts with Haversine distance to phone
  JsonArray alerts = response.createNestedArray("elephantAlerts");
  for (int i = 0; i < pillarCount; i++) {
    if (pillars[i].elephantDetected && pillars[i].active) {
      JsonObject alert = alerts.createNestedObject();
      alert["pillarId"]   = pillars[i].id;
      alert["pillarName"] = pillars[i].name;
      alert["lat"]        = pillars[i].lat;
      alert["lon"]        = pillars[i].lon;
      alert["count"]      = pillars[i].elephantCount;
      alert["adult"]      = pillars[i].elephantAdult;
      alert["calf"]       = pillars[i].elephantCalf;
      alert["behavior"]   = pillars[i].elephantBehavior;
      alert["group"]      = pillars[i].elephantGroup;
      alert["detectedAt"] = pillars[i].detectedAt;

      if (phoneGPSReceived) {
        // Haversine: phone GPS → this pillar GPS
        double dist   = haversineDistance(phoneLat, phoneLon, pillars[i].lat, pillars[i].lon);
        double distKm = dist / 1000.0;
        alert["distance_m"]  = dist;
        alert["distance_km"] = distKm;
        alert["riskLevel"]   = calculateRiskLevel(distKm, pillars[i].elephantBehavior, pillars[i].elephantGroup);
      }
    }
  }

  String output;
  serializeJson(response, output);
  server.send(200, "application/json", output);
}

// ── OPTIONS handler (CORS preflight) ────────────────────────────────────
void handleOptions() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
  server.send(204);
}

// =========================================================================
// sendAck() — Send acknowledgement back to sender
// =========================================================================
void sendAck(uint16_t seq) {
  String ack = "ACK:" + String(seq);
  LoRa.beginPacket();
  LoRa.print(ack);
  LoRa.endPacket();
  // Switch back to receive mode immediately
  LoRa.receive();
}

// =========================================================================
// parseAndDisplay() — Parse JSON payload, print formatted results, store
// =========================================================================
void parseAndDisplay(const char* jsonStr, int rssi, float snr) {
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, jsonStr);

  if (err) {
    Serial.printf("  [ERROR] JSON parse failed: %s\n", err.c_str());
    Serial.printf("  Raw data: %s\n", jsonStr);
    return;
  }

  bool elephant = doc["elephant"] | false;

  // ── Update detection state ──────────────────────────────────────────
  latestDetection.elephantDetected = elephant;
  latestDetection.rssi = rssi;
  latestDetection.snr  = snr;

  if (elephant) {
    elephantAlerts++;

    latestDetection.count = doc["count"] | 0;
    latestDetection.adult = doc["adult"] | 0;
    latestDetection.calf  = doc["calf"]  | 0;
    latestDetection.lat   = doc["lat"]   | 0.0;
    latestDetection.lon   = doc["lon"]   | 0.0;

    strlcpy(latestDetection.pillarId,  doc["pillar_id"] | "?", sizeof(latestDetection.pillarId));
    strlcpy(latestDetection.group,     doc["group"]     | "?", sizeof(latestDetection.group));
    strlcpy(latestDetection.behavior,  doc["behavior"]  | "?", sizeof(latestDetection.behavior));
    strlcpy(latestDetection.timestamp, doc["ts"]        | "?", sizeof(latestDetection.timestamp));

    // Check for aggressive behavior
    bool aggressive = (strcmp(latestDetection.behavior, "aggressive") == 0);
    if (aggressive) {
      aggressiveAlerts++;
    }

    // ── Print elephant detection details ──────────────────────────────
    Serial.println("  ╔══════════════════════════════════════════╗");
    Serial.println("  ║     *** ELEPHANT DETECTED ***            ║");
    Serial.println("  ╠══════════════════════════════════════════╣");
    Serial.printf("  ║ Pillar ID  : %-27s ║\n", latestDetection.pillarId);
    Serial.printf("  ║ Total Count: %-27d ║\n", latestDetection.count);
    Serial.printf("  ║ Adults     : %-27d ║\n", latestDetection.adult);
    Serial.printf("  ║ Calves     : %-27d ║\n", latestDetection.calf);
    Serial.printf("  ║ Group Type : %-27s ║\n", latestDetection.group);

    if (aggressive) {
      Serial.println("  ║ Behavior   : *** AGGRESSIVE ***          ║");
    } else {
      Serial.printf("  ║ Behavior   : %-27s ║\n", latestDetection.behavior);
    }

    Serial.printf("  ║ Latitude   : %-27.6f ║\n", latestDetection.lat);
    Serial.printf("  ║ Longitude  : %-27.6f ║\n", latestDetection.lon);
    Serial.printf("  ║ Timestamp  : %-27s ║\n", latestDetection.timestamp);
    Serial.println("  ╠══════════════════════════════════════════╣");
    Serial.printf("  ║ RSSI       : %-22d dBm ║\n", rssi);
    Serial.printf("  ║ SNR        : %-22.1f dB  ║\n", snr);
    Serial.println("  ╚══════════════════════════════════════════╝");

    // Print Haversine distance to phone if GPS is available
    if (phoneGPSReceived) {
      double dist = haversineDistance(phoneLat, phoneLon,
                                       latestDetection.lat, latestDetection.lon);
      Serial.printf("  [HAVERSINE] Phone ↔ Pillar %s = %.2f m (%.2f km)\n",
                    latestDetection.pillarId, dist, dist / 1000.0);
    }

    // ── Activate buzzer for aggressive behavior ──────────────────────
    if (aggressive) {
      Serial.println("  >>> BUZZER ACTIVATED — AGGRESSIVE BEHAVIOR <<<");
      for (int i = 0; i < 5; i++) {
        digitalWrite(BUZZER_PIN, HIGH); delay(100);
        digitalWrite(BUZZER_PIN, LOW);  delay(100);
      }
    } else {
      // Single short beep for calm detection
      digitalWrite(BUZZER_PIN, HIGH); delay(200);
      digitalWrite(BUZZER_PIN, LOW);
    }

    // LED stays on for 1 second on detection
    digitalWrite(LED_PIN, HIGH); delay(1000); digitalWrite(LED_PIN, LOW);

  } else {
    // ── No elephant ──────────────────────────────────────────────────
    const char* pillar = doc["pillar_id"] | "?";
    strlcpy(latestDetection.pillarId, pillar, sizeof(latestDetection.pillarId));

    Serial.println("  ┌──────────────────────────────────────────┐");
    Serial.printf("  │ Pillar     : %-27s │\n", pillar);
    Serial.println("  │ Elephant   : NO                          │");
    Serial.printf("  │ RSSI       : %-22d dBm │\n", rssi);
    Serial.printf("  │ SNR        : %-22.1f dB  │\n", snr);
    Serial.println("  └──────────────────────────────────────────┘");

    // Quick LED blink = status received, no elephant
    digitalWrite(LED_PIN, HIGH); delay(100); digitalWrite(LED_PIN, LOW);
  }

  // Store in pillar memory for app access via WebServer
  storePillarAlert(jsonStr);
}

// =========================================================================
// checkLoRa() — Listen for LoRa packets, send ACK, display results
// =========================================================================
void checkLoRa() {
  int packetSize = LoRa.parsePacket();
  if (packetSize == 0) return;

  // ── Reject packets that are too small or too large ────────────────────
  if (packetSize < 10 || packetSize > 500) {
    Serial.printf("  [REJECT] LoRa packet size out of range: %d bytes\n", packetSize);
    // Drain the packet
    while (LoRa.available()) LoRa.read();
    return;
  }

  // ── Read the full packet ──────────────────────────────────────────────
  String incoming = "";
  incoming.reserve(packetSize + 1);
  while (LoRa.available()) {
    incoming += (char)LoRa.read();
  }

  int   rssi = LoRa.packetRssi();
  float snr  = LoRa.packetSnr();
  totalReceived++;
  loraLastReceivedTime = millis();

  Serial.println();
  Serial.println("═══════════════════════════════════════════════");
  Serial.printf("[PACKET #%lu] Received via LoRa (%d bytes, RSSI=%d, SNR=%.1f)\n",
                totalReceived, packetSize, rssi, snr);
  Serial.println("───────────────────────────────────────────────");

  // ── Reject very weak signals (unreliable data) ────────────────────────
  if (rssi < -130) {
    Serial.printf("  [WARN] Very weak signal (RSSI=%d) — data may be corrupted\n", rssi);
  }

  // ── Parse "SEQ:xxxx|<json>" format ────────────────────────────────────
  uint16_t    seq       = 0;
  const char* jsonStart = NULL;

  int pipeIndex = incoming.indexOf('|');
  if (incoming.startsWith("SEQ:") && pipeIndex > 4) {
    seq       = (uint16_t)incoming.substring(4, pipeIndex).toInt();
    jsonStart = incoming.c_str() + pipeIndex + 1;

    // ── Send ACK ──────────────────────────────────────────────────────
    sendAck(seq);
    Serial.printf("  [ACK] Sent for seq=%u\n", seq);

    // ── Check for duplicate ───────────────────────────────────────────
    if (seq == lastSeq) {
      Serial.printf("  [INFO] Duplicate seq=%u — ACK sent, skipping display\n", seq);
    } else {
      lastSeq = seq;
      parseAndDisplay(jsonStart, rssi, snr);
    }
  } else {
    // Non-standard format — try to parse as raw JSON
    Serial.println("  [WARN] Non-standard packet format, trying raw parse");
    parseAndDisplay(incoming.c_str(), rssi, snr);
  }

  // ── Stats ─────────────────────────────────────────────────────────────
  Serial.println("───────────────────────────────────────────────");
  Serial.printf("  Stats: Total=%lu  Elephants=%lu  Aggressive=%lu\n",
                totalReceived, elephantAlerts, aggressiveAlerts);
  Serial.println("═══════════════════════════════════════════════");
}

// =========================================================================
// WiFi auto-reconnect (STA mode only)
// =========================================================================
void checkWiFiReconnect() {
  if (WiFi.getMode() != WIFI_STA) return;  // skip in AP mode
  if (WiFi.status() == WL_CONNECTED) return;  // already connected
  if ((millis() - lastWiFiReconnectAttempt) < WIFI_RECONNECT_MS) return;

  lastWiFiReconnectAttempt = millis();
  Serial.println("[WIFI] Connection lost — attempting reconnect...");
  WiFi.disconnect();
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 10) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WIFI] Reconnected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("[WIFI] Reconnect failed — will retry...");
  }
}

// =========================================================================
// printPeriodicStatus() — Print distance summary to Serial every 30s
// =========================================================================
void printPeriodicStatus() {
  if (millis() - lastStatusPrintTime < STATUS_PRINT_INTERVAL) return;
  lastStatusPrintTime = millis();

  // Count active elephants
  int activeElephants = 0;
  for (int i = 0; i < pillarCount; i++) {
    if (pillars[i].elephantDetected && pillars[i].active) activeElephants++;
  }

  Serial.println();
  Serial.println("─── PERIODIC STATUS ───");
  Serial.printf("  LoRa packets     : %lu | Elephant alerts: %lu\n", totalReceived, elephantAlerts);
  Serial.printf("  Active elephants : %d | Pillars in RAM: %d\n", activeElephants, pillarCount);

  if (phoneGPSReceived) {
    unsigned long gpAge = (millis() - lastGPSUpdateTime) / 1000;
    bool gpsFresh = isPhoneGPSFresh();
    Serial.printf("  Phone GPS        : %.6f, %.6f (acc: %.1fm, age: %lus%s)\n",
                  phoneLat, phoneLon, phoneGPSAccuracy, gpAge,
                  gpsFresh ? "" : " [STALE]");

    if (cachedNearestDistance >= 0) {
      Serial.printf("  Nearest elephant : %.1f m (%.2f km) — risk: %s — pillar: %s\n",
                    cachedNearestDistance,
                    cachedNearestDistance / 1000.0,
                    cachedRiskLevel.c_str(),
                    cachedNearestPillarId.c_str());
    }
  } else {
    Serial.println("  Phone GPS        : NOT YET RECEIVED (waiting for app...)");
  }

  String ip = (WiFi.getMode() == WIFI_AP) ? WiFi.softAPIP().toString() : WiFi.localIP().toString();
  Serial.printf("  WiFi IP          : %s (RSSI: %d dBm)\n", ip.c_str(), WiFi.RSSI());
  Serial.println("───────────────────────");
}

// =========================================================================
// setup()
// =========================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.println();
  Serial.println("=============================================");
  Serial.println(" ESP32 LoRa RECEIVER — Elephant Detection");
  Serial.println(" + WiFi WebServer + Haversine Distance");
  Serial.println("=============================================");
  Serial.println();

  // ── Initialise LoRa ───────────────────────────────────────────────────
  SPI.begin(18, 19, 23, LORA_SS);
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);

  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("[ERROR] LoRa init failed! Check wiring.");
    while (true) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(200);
    }
  }

  LoRa.setSpreadingFactor(LORA_SPREAD_FACTOR);
  LoRa.setSignalBandwidth(LORA_BANDWIDTH);
  LoRa.setCodingRate4(LORA_CODING_RATE);
  LoRa.setTxPower(LORA_TX_POWER);
  LoRa.setSyncWord(LORA_SYNC_WORD);
  LoRa.enableCrc();

  Serial.println("[OK] LoRa initialised");
  Serial.print("  Frequency : "); Serial.print(LORA_FREQUENCY / 1E6, 1); Serial.println(" MHz");
  Serial.print("  SF        : "); Serial.println(LORA_SPREAD_FACTOR);
  Serial.print("  BW        : "); Serial.print(LORA_BANDWIDTH / 1E3, 0); Serial.println(" kHz");
  Serial.print("  Sync Word : 0x"); Serial.println(LORA_SYNC_WORD, HEX);

  // ── Connect to WiFi ───────────────────────────────────────────────────
  Serial.println("\nConnecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    Serial.print(".");
    delay(500);
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[OK] WiFi Connected!");
    Serial.printf("  SSID       : %s\n", ssid);
    Serial.printf("  IP Address : %s\n", WiFi.localIP().toString().c_str());
    Serial.println("\n=============================================");
    Serial.printf("  Mobile App → http://%s/gps\n", WiFi.localIP().toString().c_str());
    Serial.println("=============================================\n");
  } else {
    WiFi.mode(WIFI_AP);
    WiFi.softAP("ESP32-Elephant", "12345678");
    Serial.println("\n[OK] AP Mode Started");
    Serial.println("  SSID     : ESP32-Elephant");
    Serial.println("  Password : 12345678");
    Serial.printf("  IP Address : %s\n", WiFi.softAPIP().toString().c_str());
    Serial.println("\n=============================================");
    Serial.printf("  Mobile App → http://%s/gps\n", WiFi.softAPIP().toString().c_str());
    Serial.println("=============================================\n");
  }

  // ── Setup WebServer routes ────────────────────────────────────────────
  server.on("/",       HTTP_GET,  handleRoot);
  server.on("/gps",    HTTP_POST, handleGPS);      // Phone sends GPS here
  server.on("/status", HTTP_GET,  handleStatus);

  server.on("/gps",    HTTP_OPTIONS, handleOptions);
  server.on("/status", HTTP_OPTIONS, handleOptions);

  server.enableCORS(true);
  server.begin();

  Serial.println("Listening for LoRa packets from sender...");
  Serial.println("Waiting for phone GPS via POST /gps ...");
  Serial.println("─────────────────────────────────────────────");

  // Three quick blinks = receiver ready
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH); delay(100);
    digitalWrite(LED_PIN, LOW);  delay(100);
  }
}

// =========================================================================
// loop()
// =========================================================================
void loop() {
  server.handleClient();   // Handle phone GPS requests
  checkLoRa();              // Listen for LoRa packets from sender
  checkWiFiReconnect();     // Auto-reconnect WiFi if dropped
  expireStalePillars();     // Mark stale pillar data as inactive
  printPeriodicStatus();    // Print status every 30s
}