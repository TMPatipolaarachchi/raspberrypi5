/*
 * ============================================================================
 * ESP32 LoRa RECEIVER — Elephant Detection System
 * ============================================================================
 *
 * Listens for LoRa packets sent by the sender sketch (format: "SEQ:xxxx|<json>")
 * Parses the sequence number and JSON payload, prints a human-readable summary,
 * and replies with "ACK:xxxx" so the sender can confirm delivery.
 *
 * LoRa wiring: same pinout as sender (see README3.md)
 * Required Arduino libraries:
 *   - LoRa by Sandeep Mistry
 *   - ArduinoJson by Benoit Blanchon
 */

#include <SPI.h>
#include <LoRa.h>
#include <ArduinoJson.h>

#define LORA_SS    5
#define LORA_RST   14
#define LORA_DIO0  26

#define LORA_FREQUENCY 433E6
#define LED_PIN 2

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println();
  Serial.println("=============================================");
  Serial.println(" ESP32 LoRa RECEIVER - Elephant Detection");
  Serial.println("=============================================");

  SPI.begin(18, 19, 23, LORA_SS);
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);

  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("[ERROR] LoRa init failed! Check wiring.");
    while (true) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(200);
    }
  }

  LoRa.setSpreadingFactor(7);
  LoRa.setSignalBandwidth(125E3);
  LoRa.setSyncWord(0x12);
  LoRa.enableCrc();

  Serial.println("[OK] LoRa initialised - waiting for packets...");
}

void printPayloadSummary(const char* jsonStr) {
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, jsonStr);

  Serial.println("  ┌──────────────────────────────────────┐");

  if (err) {
    Serial.println("  │ [Could not parse JSON for summary]   │");
    Serial.println("  └──────────────────────────────────────┘");
    return;
  }

  bool elephant = doc["elephant"] | false;
  const char* pillar = doc["pillar_id"] | "?";

  Serial.printf("  │ Pillar     : %-24s │\n", pillar);

  if (elephant) {
    int adult = doc["adult_count"] | 0;
    int calf = doc["calf_count"] | 0;
    int total = doc["elephant_count"] | (adult + calf);
    const char* group = doc["group_type"] | "?";
    const char* behavior = doc["behavior"] | "?";
    const char* ts = doc["timestamp"] | "?";

    Serial.println("  │ Elephant   : YES                      │");
    Serial.printf("  │ Adults     : %-24d │\n", adult);
    Serial.printf("  │ Calves     : %-24d │\n", calf);
    Serial.printf("  │ Total      : %-24d │\n", total);
    Serial.printf("  │ Group      : %-24s │\n", group);
    Serial.printf("  │ Behavior   : %-24s │\n", behavior);
    Serial.printf("  │ Timestamp  : %-24s │\n", ts);
  } else {
    Serial.println("  │ Elephant   : NO                       │");
  }

  Serial.println("  └──────────────────────────────────────┘");
}

void loop() {
  int packetSize = LoRa.parsePacket();
  if (packetSize > 0) {
    String packet = "";
    while (LoRa.available()) {
      packet += (char)LoRa.read();
    }

    Serial.println();
    Serial.print("[RX] "); Serial.println(packet);

    // Expected format: SEQ:1234|{...}
    int sep = packet.indexOf('|');
    uint16_t seq = 0;
    String json = "";

    if (packet.startsWith("SEQ:") && sep > 4) {
      seq = (uint16_t)packet.substring(4, sep).toInt();
      json = packet.substring(sep + 1);
    } else {
      // If no SEQ header, treat entire packet as JSON
      json = packet;
    }

    printPayloadSummary(json.c_str());

    // Send ACK back with same sequence number
    String ack = "ACK:" + String(seq);
    LoRa.beginPacket();
    LoRa.print(ack);
    LoRa.endPacket();

    Serial.print("[ACK] Sent -> "); Serial.println(ack);

    // Blink LED to show packet received
    digitalWrite(LED_PIN, HIGH);
    delay(80);
    digitalWrite(LED_PIN, LOW);
  }

  delay(10);
}
