# README4 - ESP32 LoRa Receiver for This Project

This file contains the ESP32 LoRa receiver code matched to the sender in `README3.md` and instructions to upload it.

## Purpose
- Listens for LoRa packets from the sender sketch. The sender transmits packets in the format: `SEQ:<seq>|<json>`.
- Parses the sequence number and JSON payload, prints a human-readable summary, and replies with `ACK:<seq>` so the sender can confirm delivery.

## Wiring (same as sender)
- VCC -> 3.3V
- GND -> GND
- SCK -> GPIO 18
- MISO -> GPIO 19
- MOSI -> GPIO 23
- NSS (CS) -> GPIO 5
- RST -> GPIO 14
- DIO0 -> GPIO 26

## Libraries Required
- LoRa by Sandeep Mistry
- ArduinoJson by Benoit Blanchon

## Receiver Sketch

```cpp
// See file: esp32_lora_receiver.ino
// (Full sketch included below for easy copy/paste)

/*
 * ESP32 LoRa RECEIVER — Elephant Detection System
 * Listens for "SEQ:<seq>|<json>", prints summary, and replies "ACK:<seq>".
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

void loop() {
  int packetSize = LoRa.parsePacket();
  if (packetSize > 0) {
    String packet = "";
    while (LoRa.available()) {
      packet += (char)LoRa.read();
    }

    Serial.println();
    Serial.print("[RX] "); Serial.println(packet);

    int sep = packet.indexOf('|');
    uint16_t seq = 0;
    String json = "";

    if (packet.startsWith("SEQ:") && sep > 4) {
      seq = (uint16_t)packet.substring(4, sep).toInt();
      json = packet.substring(sep + 1);
    } else {
      json = packet;
    }

    // Print JSON summary (pillar_id, elephant, adult_count, calf_count, etc.)
    StaticJsonDocument<512> doc;
    DeserializationError err = deserializeJson(doc, json);
    if (err) {
      Serial.println("[WARN] JSON parse failed for payload");
    } else {
      const char* pillar = doc["pillar_id"] | "?";
      bool elephant = doc["elephant"] | false;
      Serial.printf("Pillar: %s  Elephant: %s\n", pillar, elephant ? "YES" : "NO");
    }

    // Send ACK
    String ack = "ACK:" + String(seq);
    LoRa.beginPacket();
    LoRa.print(ack);
    LoRa.endPacket();

    Serial.print("[ACK] Sent -> "); Serial.println(ack);

    digitalWrite(LED_PIN, HIGH);
    delay(80);
    digitalWrite(LED_PIN, LOW);
  }

  delay(10);
}
```

## Upload Steps
1. Open Arduino IDE.
2. Select board: `ESP32 Dev Module`.
3. Install libraries: `LoRa` and `ArduinoJson`.
4. Open `esp32_lora_receiver.ino` (file in project root).
5. Set correct LoRa frequency for your region (433/868/915). Upload.

## Testing
- Start the receiver on one ESP32.
- Connect the other ESP32 (sender) to the Raspberry Pi and run the pipeline; the sender will forward JSON over LoRa and request ACKs.
- The receiver should print each packet and reply with `ACK:<seq>`.

---

File in repo: `esp32_lora_receiver.ino` (same directory as this README)
