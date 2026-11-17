// Inférence du CNN 1D de gestures/model.py (BatchNorm déjà fusionnée dans les poids).
// Architecture : Conv(6->16,k5) ReLU MaxPool2 | Conv(16->32,k5) ReLU MaxPool2 |
//                Conv(32->64,k3) ReLU | moyenne globale | Linear(64->3)
#pragma once
#include <math.h>
#include "model_weights.h"

#define N_CH 6
#define N_SAMPLES 75  // 1,5 s à 50 Hz

// Convolution 1D "same" (padding = k/2) + ReLU. in : [cin][len], out : [cout][len]
static void conv1d_relu(const float* in, int cin, int len, const float* w, const float* b,
                        int cout, int k, float* out) {
  const int pad = k / 2;
  for (int o = 0; o < cout; o++) {
    for (int t = 0; t < len; t++) {
      float s = b[o];
      for (int i = 0; i < cin; i++) {
        const float* wi = w + (o * cin + i) * k;
        const float* xi = in + i * len;
        for (int j = 0; j < k; j++) {
          int p = t + j - pad;
          if (p >= 0 && p < len) s += wi[j] * xi[p];
        }
      }
      out[o * len + t] = s > 0.f ? s : 0.f;
    }
  }
}

// MaxPool1d(2) : longueur de sortie = len / 2 (arrondi inférieur, comme PyTorch)
static int maxpool2(const float* in, int c, int len, float* out) {
  const int lo = len / 2;
  for (int i = 0; i < c; i++)
    for (int t = 0; t < lo; t++) {
      float a = in[i * len + 2 * t], b = in[i * len + 2 * t + 1];
      out[i * lo + t] = a > b ? a : b;
    }
  return lo;
}

// Normalisation "record" : z-score par axe sur la fenêtre (écart-type de population, comme numpy).
// x : [N_CH][N_SAMPLES], modifié sur place.
static void normalize_window(float* x) {
  for (int c = 0; c < N_CH; c++) {
    float* v = x + c * N_SAMPLES;
    double m = 0;
    for (int t = 0; t < N_SAMPLES; t++) m += v[t];
    m /= N_SAMPLES;
    double var = 0;
    for (int t = 0; t < N_SAMPLES; t++) var += (v[t] - m) * (v[t] - m);
    float sd = (float)sqrt(var / N_SAMPLES) + 1e-8f;
    for (int t = 0; t < N_SAMPLES; t++) v[t] = (float)((v[t] - m) / sd);
  }
}

// x : fenêtre normalisée [N_CH][N_SAMPLES]. probs : N_CLASSES probabilités (softmax).
// Renvoie l'indice de la classe prédite.
static int cnn_predict(const float* x, float* probs) {
  static float a1[16 * 75], p1[16 * 37];
  static float a2[32 * 37], p2[32 * 18];
  static float a3[64 * 18];

  conv1d_relu(x, 6, 75, CNN_W1, CNN_B1, 16, 5, a1);
  int l1 = maxpool2(a1, 16, 75, p1);           // 37
  conv1d_relu(p1, 16, l1, CNN_W2, CNN_B2, 32, 5, a2);
  int l2 = maxpool2(a2, 32, l1, p2);           // 18
  conv1d_relu(p2, 32, l2, CNN_W3, CNN_B3, 64, 3, a3);

  float feat[64];
  for (int c = 0; c < 64; c++) {
    float s = 0;
    for (int t = 0; t < l2; t++) s += a3[c * l2 + t];
    feat[c] = s / l2;
  }

  float logits[N_CLASSES], mx = -1e30f;
  for (int k = 0; k < N_CLASSES; k++) {
    float s = CNN_BF[k];
    for (int c = 0; c < 64; c++) s += CNN_WF[k * 64 + c] * feat[c];
    logits[k] = s;
    if (s > mx) mx = s;
  }
  float sum = 0;
  for (int k = 0; k < N_CLASSES; k++) { probs[k] = expf(logits[k] - mx); sum += probs[k]; }
  int best = 0;
  for (int k = 0; k < N_CLASSES; k++) {
    probs[k] /= sum;
    if (probs[k] > probs[best]) best = k;
  }
  return best;
}
