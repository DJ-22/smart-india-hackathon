# Demo generator baseline

Checkpoint: `ml/checkpoints/l1_run4`
Channel: Opus round-trip (as it would arrive over LiveKit)
Bands: AMBER >= 0.555, RED >= 0.700, EMA alpha 0.35, 4 s window / 1 s hop

| file | truth | dur s | windows | mean P(fake) | peak EMA | band |
|---|---|---|---|---|---|---|
| 1.wav | fake | 10.8 | 7 | 0.001 | 0.001 | GREEN |
| 2.wav | fake | 21.4 | 18 | 0.063 | 0.355 | GREEN |
| clip_01.wav | fake | 4.1 | 1 | 0.974 | 0.974 | RED |
| clip_02.wav | fake | 4.8 | 1 | 0.997 | 0.997 | RED |
| clip_03.wav | fake | 3.9 | 1 | 0.916 | 0.916 | RED |
| clip_04.wav | fake | 4.8 | 1 | 0.955 | 0.955 | RED |
| clip_05.wav | fake | 4.3 | 1 | 0.996 | 0.996 | RED |
| clip_06.wav | fake | 4.0 | 1 | 0.708 | 0.708 | RED |
| clip_07.wav | fake | 4.7 | 1 | 0.993 | 0.993 | RED |
| clip_08.wav | fake | 3.8 | 1 | 0.987 | 0.987 | RED |
| clip_09.wav | fake | 4.2 | 1 | 0.910 | 0.910 | RED |
| clip_10.wav | fake | 4.2 | 1 | 0.954 | 0.954 | RED |
| clip_11.wav | fake | 4.2 | 1 | 0.935 | 0.935 | RED |
| clip_12.wav | fake | 5.0 | 2 | 0.501 | 0.906 | RED |
| clip_13.wav | fake | 4.7 | 1 | 0.993 | 0.993 | RED |
| clip_14.wav | fake | 4.3 | 1 | 0.931 | 0.931 | RED |
| clip_15.wav | fake | 4.6 | 1 | 0.996 | 0.996 | RED |
| clip_16.wav | fake | 4.5 | 1 | 0.993 | 0.993 | RED |
| clip_17.wav | fake | 5.0 | 1 | 0.926 | 0.926 | RED |
| clip_18.wav | fake | 4.4 | 1 | 0.995 | 0.995 | RED |
| clip_19.wav | fake | 4.4 | 1 | 0.004 | 0.004 | GREEN |
| clip_20.wav | fake | 3.8 | 1 | 0.795 | 0.795 | RED |
| 1.wav | real | 10.6 | 7 | 0.014 | 0.023 | GREEN |
| 2.wav | real | 13.6 | 10 | 0.336 | 0.516 | GREEN |
| 3.wav | real | 16.1 | 13 | 0.338 | 0.509 | GREEN |
| 4.wav | real | 11.7 | 8 | 0.335 | 0.629 | AMBER |
| 5.wav | real | 16.4 | 13 | 0.146 | 0.448 | GREEN |

## Verdict

Cloned clips flagged RED: **86%** (19/22), mean peak EMA 0.828
Genuine clips false-alarmed RED: **0%** (0/5), mean peak EMA 0.425
EER on this set: **6.82%**, AUC 0.8727

**Action:** this generator is caught -- no extra training data needed.
