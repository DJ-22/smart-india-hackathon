# ml/ — Person A runbook

Everything A owns. Run from the repo root with the venv active.

This is exactly what produced the shipped model (`l1_run4`):

```bash
python ml/fetch_data.py harvest         # index all 31,102 train rows (~10 min, resumable)
python ml/fetch_data.py refresh         # re-sign the urls if the index is >1 day old
python ml/fetch_data.py download 16000  # stratified WAV cache + manifest.csv (~25 min, 3.3 GB)
python ml/data.py                       # print splits + leak check
python ml/augment.py                    # verify Opus and reverb are not no-ops
python ml/train_l1.py --run run3 --epochs 3 --lr 1.5e-5 --bs 16 --accum 1 --workers 0
python ml/train_l1.py --run run4 --base ml/checkpoints/l1_run3 \
                      --epochs 2 --lr 1e-5 --bs 16 --accum 1 --workers 0
python ml/train_l0.py                   # L0 gate + T_LOW / T_HIGH sweep
python ml/evaluate.py   --l1 ml/checkpoints/l1_run4 --baseline   # -> RESULTS.md
python ml/robustness.py --l1 ml/checkpoints/l1_run4              # channel sweep
python ml/calibrate.py  --l1 ml/checkpoints/l1_run4 --write      # H8:00, writes config.py
python ml/export.py     --l1 ml/checkpoints/l1_run4              # -> checkpoints/MANIFEST.json
```

run4 warm-starts from run3 rather than the base checkpoint: it only has to learn
the room, not the task, so it converges in ~17 min instead of ~30.

`--workers 0` is not laziness: spawned dataloader workers each map the CUDA DLLs
and blew the Windows commit limit (`WinError 1455, the paging file is too small`).

Run history, all compared on the identical split — adopt a run only if it wins here:

| run | data | recipe | val EER | unseen-lang (Tamil) | new corpus |
|---|---|---|---|---|---|
| run1 | 5.5k | 3 @ 2e-5 | 7.14% | 4.20% | 29.00% |
| run2 | 5.5k | 4 @ 1.5e-5 | 5.43% | 3.80% | 30.05% |
| run3 | 11k | 3 @ 1.5e-5 | 4.35% | 2.80% | 23.00% |
| **run4** | **11k** | **+ reverb aug, warm start** | **see below** | | |

Doubling the data beat every hyperparameter change (run2 → run3). Adding reverb
augmentation (run3 → run4) is what made the model usable on real recordings —
it barely moved clean val EER but transformed everything else:

| | run3 | run4 |
|---|---|---|
| clean val EER | 1.67% | **0.83%** |
| EER, reverb T60 0.30 s | 54.17% | **15.00%** |
| P(fake) on *genuine* clips, reverb 0.30 s | 0.877 | **0.197** |
| real laptop recordings false-alarmed RED | **5 / 5** | **0 / 5** |
| real demo set (22 cloned + 5 genuine) | AUC 0.35 | **AUC 0.87** |

## Testing your own clips

Drop audio in `$VG_DATA_DIR/demo_clips/fake/` and `real/` (any format, rate,
or length), then:

```bash
python ml/demo_baseline.py            # defaults to the shipped checkpoint
```

It resamples to 16 kHz, slides the server's real window over each clip, and
reports the GREEN/AMBER/RED the live UI would show. Two clips is not a
measurement — an early read on two files said "this generator is undetectable",
and 20 clips from the same tool said 86% caught at 6.8% EER. Use ~20.

That is the one number nothing here can substitute for: everything below is
measured against IndicTTS renderings, not against the generator on stage.

## Where the data lives

`VG_DATA_DIR` in `.env` points the cache outside the repo (this repo sits in a
OneDrive folder; several GB of WAVs syncing during a sprint is not a good time).
Default is `<repo>/data` if the variable is unset.

## Things that are easy to get wrong here

**The dataset is bigger than the HF stats API claims.** `/statistics` reports
8,890 train rows; the real count is **31,102** across **16 languages**. The
parquet export is ~18 GB because the source audio is 44.1 kHz, so `fetch_data.py`
pulls per-row WAVs from the datasets-server row API and resamples to 16 kHz
instead of streaming parquet.

**Label polarity is inverted upstream.** `MelodyMachine/Deepfake-audio-detection-V2`
ships `id2label = {0: 'fake', 1: 'real'}`; our manifest uses `label 1 = fake/TTS`.
`train_l1.py` swaps the two rows of the pretrained classifier head so that after
fine-tuning, index 1 really means fake everywhere in the repo. `ml/infer.py`
resolves the fake index by name for any checkpoint, so never assume index 1 when
loading a checkpoint you did not train.

**The zero-shot baseline is not a usable fallback.** Measured on 2,055 clips,
the un-finetuned checkpoint returns P(fake) ≈ 0 for essentially everything
(median 0.000, mean 0.031 on real vs 0.027 on fake) for an overall AUC of
**0.4886** — chance. BUILD_PLAN §8 lists "ship zero-shot MelodyMachine" as the
fallback if the fine-tune fails; it is not one. The fine-tune is the product.

**Codec augmentation is real, not approximated.** libsndfile 1.2.2 encodes
Ogg/Opus, so `ml/augment.py` does a genuine Opus round-trip in memory in ~8 ms,
no ffmpeg and no torchaudio. `python ml/augment.py` asserts it actually changes
the signal — `torchaudio.functional.apply_codec`, which BUILD_PLAN §4.4 suggests,
was removed from torchaudio and would have silently no-opped.

**Reverb augmentation matters more than the codec, and the plan has no
equivalent of it.** The corpus is close-mic studio audio for *both* classes, so
"a genuine voice in a room" never appears in training. On the un-reverbed model
a T60 of 0.15 s pushed genuine speech from P(fake) 0.00 to 0.84, and 0.30 s to
0.97 — every person who spoke into a laptop was confidently called a deepfake,
which is the entire live demo. `ml/robustness.py` sweeps this; run it after any
retrain. It is the check that would have caught the problem before a judge did.

**Always resample user audio.** Everything in `data/clips` is already 16 kHz
because `fetch_data.py` resampled it, so a scorer that ignores the file's rate
looks perfect on our own data and silently mis-scores anything brought in from
outside. `ml/infer.py: read_audio` is the one place that does this; use it
rather than calling `sf.read` directly.

**L0 is fit on codec-augmented audio only**, and its threshold is tuned for safe
discharge (largest `T_LOW` keeping ≥99% of fakes in play), never for F1.

**The cascade does not pay off on this hardware, and `CASCADE_ENABLED` is False.**
Three separate measurements say so, and each is in RESULTS.md:

1. L0 *can* discharge 8.6% of windows at 99% fake recall — not the 40–60%
   BUILD_PLAN §4.6 assumed, but real.
2. A discharged window reports L0's score instead of L1's, which took val EER
   from **4.35% to 8.64%**. `ml/calibrate.py` measures each shortcut against
   flat L1 and switches off any that costs more than half a point, so it sets
   `T_LOW = 0.0`. The `T_HIGH` fast-track survives that test (4.40%).
3. Then latency killed the rest: L0 costs **~13 ms/window** and L1 **~14 ms** on
   this GPU. Gating a 14 ms transformer behind a 13 ms gate is a net slowdown.

None of this is wasted — the cascade is the right architecture when L1 really is
the bottleneck (CPU-only inference, or many concurrent calls per GPU). Both
levels stay wired and calibrated, so `CASCADE_ENABLED = True` is a one-line demo.
Present it as a measured trade-off, not as an unimplemented feature.

**Benchmark latency on real speech.** The first version of the latency benchmark
fed white noise, `librosa.yin` degenerated on aperiodic input, and L0 looked like
67 ms/window instead of 13 — the difference between "the gate is cheap" and "the
gate costs more than the model".

**Sweep thresholds on score quantiles, not a fixed grid.** A well-separated
LightGBM piles most of its mass below 0.01, so the `np.arange(0.02, 0.40, 0.01)`
sweep in BUILD_PLAN §4.6 steps over every usable threshold and reports that no
safe gate exists when one does. `ml/metrics.py` sweeps quantiles instead, and
returns `None` rather than a guessed constant when no threshold is safe.

**Accuracy tracks the corpus, not the language.** The challenge set merges at
least four collections with different recording chains (`ml/data.py: source_of`).
An unseen *language* from a well-represented corpus scores 2.8% EER; an unseen
language from a corpus with 66 training clips scores 23%. Report both halves —
the combined "unseen-language EER" understates generalisation across languages
and overstates it across generators.

**`WINDOW_S` is 4.0, not the 3.0 the plan started with.** The L1 crop is 4 s, and
a 3 s window costs 1.7 points of EER purely from the train/serve mismatch. `HOP_S`
is unchanged, so the UI still updates every second.

**The cached asset URLs expire in about a day.** `data/index.jsonl` holds signed
links; a stale index fails every download with HTTP 403. Run
`python ml/fetch_data.py refresh` to append fresh ones — `load_index()` takes the
newest entry per row.

## Security notes — read before wiring `server/detector.py`

**No pickle crosses a machine boundary.** BUILD_PLAN §4.8 names the gate
`l0.pkl`; it is now **`l0.txt`** (LightGBM's native text format) plus `l0.json`
for the metadata, and `MANIFEST.json` says so (`"l0_format": "lightgbm-text"`).
Load it with `ml.evaluate.load_l0()`, which returns the same `predict_proba`
interface — verified bit-identical to the old pickle on 4,931 rows. A pickle is
arbitrary code execution for whoever unpickles it; a model_file is a list of
trees. `training_args.bin` (also a pickle, never read at inference) is no longer
shipped in the `l1_run*` directories, and the base checkpoint is safetensors.

**Cap what you decode.** `ml.infer.read_audio` refuses clips longer than
`max_s` (default 600 s) by reading the WAV *header* before decoding a sample, so
a multi-gigabyte upload is rejected for free. `/score` takes raw bytes from the
network — put the same check in front of it, with a tighter cap (a 4 s window
never needs more than a few seconds), and treat libsndfile as what it is: a C
parser fed attacker-controlled bytes. Reject anything that is not 16 kHz mono
PCM before it reaches the decoder.

**Rotate the Hugging Face token in `.env`.** The file is gitignored and was
never committed, but the token was printed to a terminal during setup. It is not
needed at all — the dataset and base model are public — so rotating it costs
nothing and removing it costs nothing either.

**Other hardening in this pass:** filenames built from dataset ids are
sanitised (`fetch_data.py`), the feature cache loads with `allow_pickle=False`,
and `calibrate.py` can only ever write a numeric literal into `config.py`.

## Environment notes

- `torch==2.11.0+cu130` + `torchaudio==2.11.0+cu130` from the cu130 index. Plain
  `pip install torch` gives a **CPU** build, and mixing index versions silently
  leaves torch on CPU while torchaudio is CUDA — check `torch.cuda.is_available()`
  after any torch reinstall.
- transformers 5.x renamed `evaluation_strategy` → `eval_strategy` and dropped
  `warmup_ratio` in favour of `warmup_steps`; `train_l1.py` handles both.
