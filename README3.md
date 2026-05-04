/*
 * ============================================================================
 * ESP32 LoRa SENDER - Elephant Detection System
 * ============================================================================
 *
 * Hardware connections:
 *   Raspberry Pi  --USB-->  ESP32 (Sender)  --LoRa-->  ESP32 (Receiver)
 *
 * This ESP32 receives JSON from the Raspberry Pi over USB Serial,
 * then forwards it to a remote ESP32 Receiver via LoRa (SX1276/SX1278).
 *
 * LoRa module wiring (SX1276/SX1278 to ESP32):
 *   LoRa Pin  ->  ESP32 Pin
 *   ------------------------
 *   VCC       ->  3.3V
 *   GND       ->  GND
 *   SCK       ->  GPIO 18
 *   MISO      ->  GPIO 19
 *   MOSI      ->  GPIO 23
 *   NSS (CS)  ->  GPIO 5
 *   RST       ->  GPIO 14
 *   DIO0      ->  GPIO 26
 *
 * Baud rate from Raspberry Pi: 115200
 * LoRa frequency: 433 MHz (change to 868/915 based on your region)
 *
 * Required Arduino Libraries:
 *   1. LoRa by Sandeep Mistry
 *   2. ArduinoJson by Benoit Blanchon
 *
 * Board: ESP32 Dev Module
 * ============================================================================
 */

#include <SPI.h>
#include <LoRa.h>
#include <ArduinoJson.h>

#define LORA_SS      5
#define LORA_RST     14
#define LORA_DIO0    26

#define LORA_FREQUENCY     433E6
#define LORA_BANDWIDTH     125E3
#define LORA_SPREAD_FACTOR  7
#define LORA_TX_POWER       20
#define LORA_SYNC_WORD      0x12

#define PI_BAUD_RATE 115200
#define LED_PIN 2

#define MAX_MSG_LEN 300
char serialBuffer[MAX_MSG_LEN];
int bufferIndex = 0;

unsigned long msgReceived = 0;
unsigned long msgSentOk   = 0;
unsigned long msgSentFail = 0;

#define ACK_TIMEOUT_MS  2000
#define MAX_RETRIES     3

uint16_t packetSeq = 0;

void setup() {
  Serial.begin(PI_BAUD_RATE);
  while (!Serial) { delay(10); }

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println();
  Serial.println("=============================================");
  Serial.println(" ESP32 LoRa SENDER - Elephant Detection");
  Serial.println("=============================================");
  Serial.println("Waiting for JSON from Raspberry Pi...");
  Serial.println();

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
  LoRa.setTxPower(LORA_TX_POWER);
  LoRa.setSyncWord(LORA_SYNC_WORD);
  LoRa.enableCrc();

  Serial.println("[OK] LoRa initialised");
  Serial.print("  Frequency : "); Serial.print(LORA_FREQUENCY / 1E6, 1); Serial.println(" MHz");
  Serial.print("  SF        : "); Serial.println(LORA_SPREAD_FACTOR);
  Serial.print("  BW        : "); Serial.print(LORA_BANDWIDTH / 1E3, 0); Serial.println(" kHz");
  Serial.print("  TX Power  : "); Serial.print(LORA_TX_POWER); Serial.println(" dBm");
  Serial.print("  Sync Word : 0x"); Serial.println(LORA_SYNC_WORD, HEX);
  Serial.println();
  Serial.println("Ready. Listening on USB Serial for Pi data...");
  Serial.println("─────────────────────────────────────────────");

  for (int i = 0; i < 2; i++) {
    digitalWrite(LED_PIN, HIGH); delay(150);
    digitalWrite(LED_PIN, LOW);  delay(150);
  }
}

bool sendWithRetry(const char* jsonPayload, uint16_t seq) {
  for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    String packet = "SEQ:" + String(seq) + "|" + String(jsonPayload);

    LoRa.beginPacket();
    LoRa.print(packet);
    LoRa.endPacket();

    Serial.printf("  [TX] Attempt %d/%d  seq=%u  (%d bytes)\n",
                  attempt, MAX_RETRIES, seq, packet.length());

    LoRa.receive();
    unsigned long start = millis();
    while (millis() - start < ACK_TIMEOUT_MS) {
      int packetSize = LoRa.parsePacket();
      if (packetSize > 0) {
        String ack = "";
        while (LoRa.available()) {
          ack += (char)LoRa.read();
        }
        if (ack.startsWith("ACK:")) {
          uint16_t ackSeq = (uint16_t)ack.substring(4).toInt();
          if (ackSeq == seq) {
            int rssi = LoRa.packetRssi();
            Serial.printf("  [ACK] Received  seq=%u  RSSI=%d dBm\n", ackSeq, rssi);
            return true;
          }
        }
      }
      delay(10);
    }
    Serial.printf("  [WARN] No ACK for seq=%u (attempt %d)\n", seq, attempt);
  }
  return false;
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
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) {
        serialBuffer[bufferIndex] = '\0';

        msgReceived++;
        packetSeq++;

        Serial.println();
        Serial.println("═══════════════════════════════════════════");
        Serial.printf("[MSG #%lu] Received from Raspberry Pi\n", msgReceived);
        Serial.println("───────────────────────────────────────────");

        printPayloadSummary(serialBuffer);

        Serial.println("  Sending via LoRa...");
        digitalWrite(LED_PIN, HIGH);

        bool ok = sendWithRetry(serialBuffer, packetSeq);

        digitalWrite(LED_PIN, LOW);

        if (ok) {
          msgSentOk++;
          Serial.println("  [OK] Delivered to receiver via LoRa");
        } else {
          msgSentFail++;
          Serial.println("  [FAIL] Could not deliver (no ACK after retries)");
        }

        Serial.println("───────────────────────────────────────────");
        Serial.printf("  Stats: Recv=%lu  Sent=%lu  Failed=%lu\n",
                      msgReceived, msgSentOk, msgSentFail);
        Serial.println("═══════════════════════════════════════════");

        bufferIndex = 0;
      }
    } else {
      if (bufferIndex < MAX_MSG_LEN - 1) {
        serialBuffer[bufferIndex++] = c;
      } else {
        Serial.println("[WARN] Serial buffer overflow — message discarded");
        bufferIndex = 0;
      }
    }
  }
}


