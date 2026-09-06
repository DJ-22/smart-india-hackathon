# fixtures

Small committed test clips (16 kHz mono s16 WAV), cut from a real phone
capture over LiveKit. Shared test inputs — score against these so server,
ml and ui all test on the same audio.

- `win3s.wav` — 3 s of clear real speech; one scoring window. Should pass the VAD gate (speech_ratio ≈ 0.97) and discharge as real.
- `sil3s.wav` — 3 s of digital silence; must be gated (`prob_fake=0.0`, `level_resolved=0`, `speech_ratio=0.0`).
- `enroll8s.wav` — 8 s clear speech from the same speaker; reference voiceprint for `POST /session/{sid}/enroll`.
