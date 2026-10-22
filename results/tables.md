### Accuracy vs Es/N0 (%)

| Es/N0 (dB) | CNN (raw I/Q) | MLP on cumulants | Cumulants + QDA | Analytic rule (genie SNR) |
|---:|---:|---:|---:|---:|
| -10 | 31.9 | 26.2 | 24.8 | 24.6 |
| -8 | 38.2 | 24.8 | 25.3 | 24.9 |
| -6 | 42.3 | 25.5 | 25.2 | 24.7 |
| -4 | 47.9 | 24.9 | 24.8 | 24.9 |
| -2 | 55.1 | 27.0 | 25.4 | 25.1 |
| 0 | 60.2 | 30.9 | 27.4 | 29.3 |
| 2 | 67.6 | 40.6 | 34.5 | 36.1 |
| 4 | 74.7 | 51.9 | 47.7 | 46.0 |
| 6 | 78.0 | 61.1 | 59.5 | 57.0 |
| 8 | 83.2 | 68.2 | 68.0 | 63.9 |
| 10 | 88.0 | 72.8 | 73.0 | 72.2 |
| 12 | 91.8 | 78.8 | 72.4 | 76.0 |
| 14 | 95.0 | 86.1 | 73.2 | 82.5 |
| 16 | 96.9 | 89.6 | 73.0 | 85.0 |
| 18 | 98.5 | 91.1 | 72.8 | 87.6 |
| 20 | 98.6 | 89.8 | 71.0 | 87.6 |
| **mean 0–20 dB** | **84.8** | **69.2** | **61.1** | **65.8** |

### Per-class accuracy at 10 dB (%)

| Method | BPSK | QPSK | 16QAM | 64QAM |
|---|---:|---:|---:|---:|
| CNN (raw I/Q) | 99.8 | 97.8 | 78.8 | 75.6 |
| MLP on cumulants | 94.2 | 91.6 | 16.2 | 89.2 |
| Cumulants + QDA | 94.2 | 82.2 | 75.0 | 40.6 |
| Analytic rule (genie SNR) | 83.8 | 76.0 | 53.0 | 76.2 |

### Trained once, tested on another channel (%)

| Test channel | CNN (raw I/Q) @10 dB / mean | MLP on cumulants @10 dB / mean | Cumulants + QDA @10 dB / mean |
|---|---:|---:|---:|
| trained channel | 88.0 / 84.8 | 72.8 / 69.2 | 73.0 / 61.1 |
| AWGN only | 89.4 / 87.3 | 73.7 / 73.7 | 77.0 / 64.1 |
| Rayleigh (no LOS) | 84.6 / 81.5 | 67.4 / 65.1 | 63.7 / 58.0 |
| fast fading (10x Doppler) | 85.2 / 83.3 | 71.0 / 64.1 | 63.2 / 55.2 |
| large CFO (10x) | 70.6 / 67.9 | 73.2 / 69.4 | 72.1 / 61.3 |
| roll-off 0.2 (unseen) | 82.9 / 79.9 | 72.7 / 68.7 | 72.6 / 61.8 |

### Trained without fading (AWGN only) (%)

| Test set | CNN (raw I/Q) @10 dB / mean | MLP on cumulants @10 dB / mean | Cumulants + QDA @10 dB / mean |
|---|---:|---:|---:|
| test: AWGN | 85.8 / 85.0 | 73.7 / 73.3 | 84.2 / 64.6 |
| test: Rician fading | 82.5 / 78.6 | 72.6 / 68.2 | 74.4 / 61.5 |

### Observation length (models trained at 1024 samples) (%)

| Samples per frame | CNN (raw I/Q) @10 dB / mean | MLP on cumulants @10 dB / mean | Cumulants + QDA @10 dB / mean |
|---|---:|---:|---:|
| 128 | 76.3 / 73.3 | 56.2 / 50.6 | 51.2 / 42.8 |
| 256 | 79.5 / 77.9 | 64.9 / 59.8 | 62.2 / 51.3 |
| 512 | 82.3 / 81.3 | 71.1 / 65.9 | 68.0 / 57.7 |
| 1024 | 88.0 / 84.8 | 72.8 / 69.2 | 73.0 / 61.1 |

### Training-set size (%)

| Training frames | CNN (raw I/Q) @10 dB / mean | MLP on cumulants @10 dB / mean | Cumulants + QDA @10 dB / mean |
|---|---:|---:|---:|
| 1000 | 45.4 / 42.0 | 69.3 / 62.5 | 72.0 / 63.7 |
| 4000 | 66.0 / 61.7 | 71.8 / 64.8 | 75.1 / 64.7 |
| 16000 | 73.5 / 69.8 | 70.8 / 66.1 | 72.0 / 61.0 |
| 64000 | 81.3 / 77.8 | 73.5 / 69.0 | 72.5 / 61.1 |

### Unseen modulation (8PSK, Es/N0 ≥ 10 dB)

* CNN labels 8PSK as: BPSK 0.1%, QPSK 99.3%, 16QAM 0.3%, 64QAM 0.3%
* CNN mean max-softmax confidence: 0.991 on 8PSK vs 0.940 on the trained classes; 98.2% of 8PSK frames get confidence > 0.9
* Out-of-distribution detection AUROC: CNN max-softmax 0.411, QDA max log-likelihood 0.960 (0.5 = chance)

### Cost

| | CNN | Cumulants + QDA |
|---|---:|---:|
| Trainable parameters | 399,108 | 144 |
| Arithmetic per frame | 52.7 M multiply-accumulates (counted) | ≈ 1 M floating-point operations (estimated; the 8192-point FFT for CFO estimation and the 4x resampling FFTs dominate) |
| CPU time per frame, batch 1 | 0.87 ms | — |
| CPU time per frame, batch 256 | 3.06 ms | 0.14 ms (features + QDA) |
| GPU (mps) per frame, batch 256 | 0.036 ms | — |

Calibration: CNN expected calibration error 0.007.
Total experiment wall-clock time: 32.0 min on mps.
