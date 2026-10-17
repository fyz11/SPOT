# SAM feature extraction benchmark

250 PhC-C2DH-U373 cell contours (8 cells x 40 frames, 520x696 frames); CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (32 logical cores); GPU: NVIDIA GeForce RTX 4070 Laptop GPU

| features | variant | seconds | ms / contour | speed-up |
|---|---|---:|---:|---:|
| shape | original | 132.09 | 528.3 | 1.0x |
| shape | fast, n_jobs=1 | 32.45 | 129.8 | 4.1x |
| shape | fast, n_jobs=8 | 4.40 | 17.6 | 30.0x |
| shape | fast, n_jobs=16 | 6.70 | 26.8 | 19.7x |
| shape | torch (cuda) | 12.63 | 50.5 | 10.5x |
| appearance | original | 19.51 | 78.0 | 1.0x |
| appearance | fast, n_jobs=1 | 5.09 | 20.4 | 3.8x |
| appearance | fast, n_jobs=8 | 2.74 | 11.0 | 7.1x |
| appearance | fast, n_jobs=16 | 3.03 | 12.1 | 6.4x |
| appearance | torch (cuda) | 1.21 | 4.8 | 16.1x |
| appearance (masked) | original | 30.21 | 120.8 | 1.0x |
| appearance (masked) | fast, n_jobs=1 | 15.40 | 61.6 | 2.0x |
| appearance (masked) | fast, n_jobs=8 | 2.49 | 10.0 | 12.1x |
| appearance (masked) | fast, n_jobs=16 | 2.70 | 10.8 | 11.2x |
| appearance (masked) | torch (cuda) | 6.82 | 27.3 | 4.4x |
| motion | original | 15.31 | 61.2 | 1.0x |
| motion | fast, n_jobs=1 | 8.80 | 35.2 | 1.7x |
| motion | fast, n_jobs=8 | 4.22 | 16.9 | 3.6x |
| motion | fast, n_jobs=16 | 4.51 | 18.0 | 3.4x |
| motion | torch (cuda) | 2.05 | 8.2 | 7.5x |
