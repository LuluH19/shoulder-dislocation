// Reconnaissance de mouvements (circle / horizontal / vertical) avec un MPU6050.
// Modèle : CNN 1D entraîné par scripts/train.py, poids exportés par scripts/export_firmware.py.
//
// Fonctionnement :
//   - lecture du MPU6050 à 50 Hz en continu
//   - dès qu'un mouvement démarre (gyro ou accélération au-dessus d'un seuil), on enregistre
//     une fenêtre de 75 mesures (1,5 s), dont PRE_SAMPLES mesures prises juste avant le déclenchement
//   - la fenêtre est normalisée (z-score par axe) puis passée au réseau
//   - le résultat est affiché sur le moniteur série (115200 bauds)
//   - envoyer 'r' sur le port série force un enregistrement (compte à rebours 3 s)
//
// Aucune bibliothèque externe : le MPU6050 est lu directement en I2C avec Wire.
// Le dossier du sketch doit contenir cnn.h et model_weights.h.

#include <Wire.h>
#include "cnn.h"

// ---- Réglages ----
const int SDA_PIN = 8;
const int SCL_PIN = 9;
const unsigned long PERIOD_US = 20000;  // 50 Hz, comme les données d'entraînement
const int PRE_SAMPLES = 10;             // mesures gardées avant le déclenchement (0,2 s)
const float GYRO_TRIGGER_DPS = 40.0;    // seuil de rotation (°/s)
const float ACC_TRIGGER_G = 0.25;       // seuil d'écart à 1 g
const float MIN_CONFIDENCE = 0.60;      // en dessous : résultat affiché comme "incertain"
const unsigned long COOLDOWN_MS = 1000; // pause après une prédiction

// Sensibilités par défaut du MPU6050 (±2 g, ±250 °/s)
const float ACC_LSB_PER_G = 16384.0;
const float GYRO_LSB_PER_DPS = 131.0;

// Registres du MPU6050 (datasheet / register map InvenSense)
const uint8_t MPU_ADDR = 0x68;  // 0x69 si la broche AD0 est reliée à VCC
const uint8_t REG_PWR_MGMT_1 = 0x6B;
const uint8_t REG_GYRO_CONFIG = 0x1B;   // 0 -> ±250 °/s
const uint8_t REG_ACCEL_CONFIG = 0x1C;  // 0 -> ±2 g
const uint8_t REG_ACCEL_XOUT_H = 0x3B;  // 14 octets : acc XYZ, température, gyro XYZ
const uint8_t REG_WHO_AM_I = 0x75;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool mpuBegin() {
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) return false;  // personne ne répond à cette adresse
  writeReg(REG_PWR_MGMT_1, 0x01);  // sortie de veille, horloge = gyro X
  delay(100);
  writeReg(REG_GYRO_CONFIG, 0x00);
  writeReg(REG_ACCEL_CONFIG, 0x00);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_WHO_AM_I);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)1);
  if (Wire.available()) {
    Serial.print("WHO_AM_I = 0x");
    Serial.println(Wire.read(), HEX);  // 0x68 pour un vrai MPU6050, autre valeur pour un clone
  }
  return true;
}

float ring[PRE_SAMPLES][N_CH];  // dernières mesures avant déclenchement
int ringPos = 0, ringCount = 0;
float window_[N_CH * N_SAMPLES];  // [axe][temps], format attendu par le réseau

enum State { IDLE, RECORDING, COOLDOWN };
State state = IDLE;
int recCount = 0;
unsigned long nextSample = 0, cooldownUntil = 0;

void readSample(float* s) {
  uint8_t b[14] = {0};
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)14);
  for (int i = 0; i < 14 && Wire.available(); i++) b[i] = Wire.read();
  int16_t ax = (b[0] << 8) | b[1], ay = (b[2] << 8) | b[3], az = (b[4] << 8) | b[5];
  int16_t gx = (b[8] << 8) | b[9], gy = (b[10] << 8) | b[11], gz = (b[12] << 8) | b[13];
  // en g et °/s ; l'unité n'a pas d'effet sur le réseau (z-score par fenêtre)
  s[0] = ax / ACC_LSB_PER_G;
  s[1] = ay / ACC_LSB_PER_G;
  s[2] = az / ACC_LSB_PER_G;
  s[3] = gx / GYRO_LSB_PER_DPS;
  s[4] = gy / GYRO_LSB_PER_DPS;
  s[5] = gz / GYRO_LSB_PER_DPS;
}

bool isMoving(const float* s) {
  float g = sqrtf(s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
  float a = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  return g > GYRO_TRIGGER_DPS || fabsf(a - 1.0f) > ACC_TRIGGER_G;
}

void putInWindow(int t, const float* s) {
  for (int c = 0; c < N_CH; c++) window_[c * N_SAMPLES + t] = s[c];
}

void startRecording(bool withPreSamples) {
  recCount = 0;
  if (withPreSamples) {
    int start = (ringPos - ringCount + PRE_SAMPLES) % PRE_SAMPLES;
    for (int i = 0; i < ringCount; i++) putInWindow(recCount++, ring[(start + i) % PRE_SAMPLES]);
  }
  state = RECORDING;
  Serial.println("Enregistrement...");
}

void classify() {
  unsigned long t0 = micros();
  normalize_window(window_);
  float probs[N_CLASSES];
  int best = cnn_predict(window_, probs);
  unsigned long dt = micros() - t0;

  Serial.print(">>> ");
  if (probs[best] < MIN_CONFIDENCE) Serial.print("incertain (");
  Serial.print(CLASS_NAMES[best]);
  if (probs[best] < MIN_CONFIDENCE) Serial.print(")");
  Serial.print("   [");
  for (int k = 0; k < N_CLASSES; k++) {
    Serial.print(CLASS_NAMES[k]);
    Serial.print(" ");
    Serial.print(probs[k] * 100, 1);
    Serial.print("%");
    if (k < N_CLASSES - 1) Serial.print(" | ");
  }
  Serial.print("]  (");
  Serial.print(dt / 1000.0, 1);
  Serial.println(" ms)");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  if (!mpuBegin()) {
    Serial.println("MPU6050 introuvable : vérifier le câblage (SDA=8, SCL=9, VCC, GND) et l'adresse (0x68/0x69).");
    while (true) delay(1000);
  }
  Serial.println("MPU6050 OK. Faire un mouvement (circle / horizontal / vertical),");
  Serial.println("ou envoyer 'r' pour un enregistrement avec compte à rebours.");
  nextSample = micros();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if ((c == 'r' || c == 'R') && state != RECORDING) {
      for (int i = 3; i > 0; i--) { Serial.println(i); delay(1000); }
      ringCount = 0;
      startRecording(false);
      nextSample = micros();
    }
  }

  if ((long)(micros() - nextSample) < 0) return;
  nextSample += PERIOD_US;

  float s[N_CH];
  readSample(s);

  switch (state) {
    case IDLE:
      if (isMoving(s)) {
        startRecording(true);
        putInWindow(recCount++, s);
      } else {
        for (int c = 0; c < N_CH; c++) ring[ringPos][c] = s[c];
        ringPos = (ringPos + 1) % PRE_SAMPLES;
        if (ringCount < PRE_SAMPLES) ringCount++;
      }
      break;

    case RECORDING:
      putInWindow(recCount++, s);
      if (recCount >= N_SAMPLES) {
        classify();
        state = COOLDOWN;
        cooldownUntil = millis() + COOLDOWN_MS;
      }
      break;

    case COOLDOWN:
      if ((long)(millis() - cooldownUntil) >= 0) {
        ringCount = 0;
        ringPos = 0;
        state = IDLE;
        Serial.println("Prêt.");
      }
      break;
  }
}
