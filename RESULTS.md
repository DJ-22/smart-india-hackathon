# VoiceGuard -- RESULTS

Model: `ml/checkpoints/l1_run4`  ·  evaluated on 4.0 s windows
Generated: 2026-09-06 05:39

## Splits

```
train  n=11061 real=5534  fake=5527  langs=Assamese,Bengali,Bodo,Dogri,English,Hindi,Kannada,Malayalam,Marathi,Nepali,Odia,Sanskrit,Telugu speakers=ASM_F,ASM_M,BEN_M,BRX_F,DOI_M,KAN_F,KAN_M,MAL_F,MAR_F,MAR_M,Malayalam_male,NEP_F,Odia_unk,SAN_M,en_f,en_m,hi_f,te_m
val    n=1932  real=966   fake=966   langs=Bengali,Bodo,Hindi,Telugu speakers=BEN_F,BRX_M,hi_m,te_f
xling  n=2999  real=1500  fake=1499  langs=Gujarati,Manipuri,Tamil speakers=Gujarati_female,Gujarati_male,Manipuri_female,Manipuri_male,TAM_F
leak check: shared speakers=none shared langs(train/xling)=none shared ids=0
```

Held-out languages (never in training): **Tamil, Gujarati, Manipuri**

## Headline numbers

Metrics slide, over the Opus channel. Hand this block to C verbatim.

```
Seen-language EER      : 4.7%   AUC 0.990
Unseen-language EER    : 15.4%   AUC 0.927   (langs: tamil, gujarati, manipuri)
  of which, same corpus : 2.4%   AUC 0.996   (tamil -- the honest
                                                 unseen-LANGUAGE number)
  of which, new corpus  : 21.8%   AUC 0.865   (gujarati, manipuri -- an unseen
                                                 RECORDING CHAIN, not just language)
Demo-generator EER     :  6.8%   AUC 0.873   (n=27: 22 cloned + 5 genuine,
                                             real recordings, our own tool)
L0 discharge rate      : 0%     (path OFF -- it could reach 8.6% but
                                cost 2.0 pts of EER; see Cascade below)
L1 transformer skipped : 7.4%   (T_HIGH fast-track only)
```

The two unseen-language rows differ by a factor of several, and the axis that separates them is the recording chain, not the language. Quoting the combined number alone would understate generalisation across languages and overstate it across generators -- both halves belong on the slide.

## Detection (L1)

| set | channel | n | EER | AUC |
|---|---|---|---|---|
| val | clean | 1932 | 2.90% | 0.9959 |
| val | opus | 1932 | 4.66% | 0.9900 |
| xling | clean | 2999 | 13.77% | 0.9404 |
| xling | opus | 2999 | 15.37% | 0.9274 |

`opus` = the same clips after a real Ogg/Opus round-trip, i.e. what the detector actually receives over LiveKit.

### Before fine-tuning

| set | channel | n | EER | AUC |
|---|---|---|---|---|
| val | opus | 1932 | 48.60% | 0.5095 |
| xling | opus | 2999 | 45.68% | 0.5770 |

`MelodyMachine/Deepfake-audio-detection-V2` off the shelf returns P(fake) near zero for essentially every clip in this corpus, so its AUC sits at chance. BUILD_PLAN section 8 lists shipping it un-finetuned as the fallback if the fine-tune fails; these rows are why that is not a fallback.

### Per language (opus channel)

| language | corpus | language seen in training | n | EER | AUC |
|---|---|---|---|---|---|
| Bengali | indictts | yes | 449 | 0.00% | 1.0000 |
| Bodo | indictts | yes | 508 | 1.18% | 0.9993 |
| Hindi | lowercase | yes | 487 | 6.16% | 0.9844 |
| Telugu | lowercase | yes | 488 | 4.72% | 0.9857 |
| Gujarati | trainset | no | 1000 | 20.20% | 0.8784 |
| Manipuri | trainset | no | 1000 | 19.70% | 0.8830 |
| Tamil | indictts | no | 999 | 2.40% | 0.9960 |

### Per corpus (opus channel) -- the axis that actually predicts accuracy

The challenge set merges several collections with different recording chains and TTS systems (see `ml/data.py: source_of`). Accuracy tracks how well a clip's *corpus* is represented in training far more than whether its *language* was seen: an unseen language from a well-represented corpus scores near the seen-language number, while an unseen language from a barely-represented corpus is several times worse.

| corpus | training clips | split | n | EER | AUC |
|---|---|---|---|---|---|
| indictts | 7971 | val | 957 | 1.25% | 0.9994 |
| lowercase | 2024 | val | 975 | 7.90% | 0.9754 |
| indictts | 7971 | xling | 999 | 2.40% | 0.9960 |
| trainset | 66 | xling | 2000 | 21.85% | 0.8648 |

## Cascade (calibrated on val, opus channel)

`T_LOW = 0.0000`, `T_HIGH = 0.9974` -- from ml/checkpoints/calibration.json (ml/calibrate.py).

**The L0 discharge path is off.** It could safely discharge 8.6% of windows at 99% fake recall, but a discharged window reports L0's score instead of L1's, which took val EER from 4.66% to 6.63% -- 2.0 points of accuracy to save 8.6% of a transformer that runs in ~14 ms. The T_HIGH fast-track path costs 0.05 points and is kept. `ml/calibrate.py` measures both paths and switches off whichever fails to pay for itself.

| set | L0 discharged | L1 (transformer) | L2 fast-tracked | fakes lost at L0 | fused EER |
|---|---|---|---|---|---|
| val_opus | 0.0% | 92.6% | 7.4% | 0.00% | 4.71% |
| xling_opus | 0.0% | 98.2% | 1.8% | 0.00% | 15.37% |

### T_LOW sweep (val, opus)

| T_LOW | fake recall kept | windows discharged |
|---|---|---|
| 0.0005 | 1.0000 | 0.0% |
| 0.0224 | 0.9948 | 6.1% |
| 0.0475 | 0.9762 | 12.1% |
| 0.0780 | 0.9565 | 18.1% |
| 0.1171 | 0.9327 | 24.1% |
| 0.1717 | 0.8986 | 30.1% |
| 0.2309 | 0.8540 | 36.1% |
| 0.3541 | 0.8116 | 42.1% |
| 0.5488 | 0.7578 | 48.1% |
| 0.7355 | 0.6977 | 54.1% |

## Demo generator (BUILD_PLAN §1)

27 clips recorded and generated by us, scored over the Opus channel with the shipped bands (`AMBER 0.645`, `RED 0.825`). This is the only number here measured against the tool we will actually demo with rather than against IndicTTS renderings.

| | cloned (n=22) | genuine (n=5) |
|---|---|---|
| flagged RED | 17 (77%) | **0** |
| flagged AMBER or RED | 19 (86%) | 0 |
| mean peak EMA | 0.828 | 0.425 |

**EER 6.82%, AUC 0.873.**

Two caveats on how far to trust it. The genuine clips are laptop recordings in a room, which is what the reverb augmentation was added for -- before it, all 5 were flagged RED and AUC was 0.35. And the cloned clips are ~4 s each, so most yield a single window and the EMA never engages; on a real call the score is averaged over dozens of windows, so 6.82% is if anything pessimistic.

Three cloned clips are missed. Two of them came from a different tool than the other twenty and score 0.001 and 0.355 -- if those represent the demo generator rather than the `clip_*` set, this number does not transfer and ~200 clips from that tool need to go into training (BUILD_PLAN §4.7).

## Latency

| level | p50 ms | p95 ms | notes |
|---|---|---|---|
| L0 (features + LightGBM) | 67.2 | 69.7 | CPU |
| L1 (wav2vec2, 4 s window) | 14.1 | 14.9 | cuda |
| L0 + L1 (worst path) | 89.1 | 95.6 | what most windows cost |

One window at a time, warm, which is how the server runs -- not batched throughput. The first call after load is ~1.8 s (lazy imports and CUDA context); `L1Scorer.warm()` exists to get that out of the way before serving, and `server/detector.py` must call it at startup.

Batched throughput for reference: L1 11.5 ms/window, L0 18.6 ms/window.

**L0's cost is a property of CPU load, not of the model.** The same code measures very differently depending on what else is resident, because librosa's STFT / MFCC / yin contend with torch's 20-thread intraop pool:

| conditions | L0 features + predict |
|---|---|
| isolated process, no torch | 12.8 ms |
| torch imported and resident | 27.0 ms |
| inside a full `evaluate.py` run (the table above) | 67.2 ms |

LightGBM is not the culprit -- tree traversal on one 76-feature row is 0.7 ms. The model is now saved with `n_jobs=1`, since single-row prediction never benefits from 28 threads and the server runs it alongside torch. Quote 12.8 ms only for an L0-only process; the server should expect the higher figures.

**The gate is not cheap relative to the model it guards.** L1 is ~14 ms on GPU regardless of load, while L0 is 13-67 ms depending on contention -- so running L0 first and L1 afterwards, which is what happens on 93% of windows, costs more than calling L1 alone under every condition measured. On a box with a GPU the cascade is a net slowdown; it pays off where L1 is genuinely the bottleneck (CPU-only inference, or many concurrent calls sharing one GPU). `CASCADE_ENABLED` is therefore False by default -- see common/config.py.

