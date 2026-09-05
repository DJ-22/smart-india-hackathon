"""Standalone diagnostic: join the LiveKit room, capture 10 s of remote
audio, write test.wav, print loud stats. NOT part of the server.

Usage (from repo root):
  python server\\capture_test.py --room demo

Loud print() is deliberate here - this is a diagnostic script.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from livekit import rtc

from server.tokens import URL, make_token

TARGET_SR = 16000
CAPTURE_S = 10.0
OUT_PATH = "test.wav"


async def _capture_track(track: rtc.Track, buf: List[np.ndarray],
                         info: dict, done: asyncio.Event) -> None:
    # Verified against livekit 1.1.17: AudioStream(track, sample_rate=...,
    # num_channels=...) resamples/downmixes SDK-side, and iterating yields
    # AudioFrameEvent objects with a .frame attribute.
    stream = rtc.AudioStream(track, sample_rate=TARGET_SR, num_channels=1)
    total = 0
    async for ev in stream:
        f = ev.frame
        if not info:
            info.update(sr=f.sample_rate, ch=f.num_channels)
            print(f"first frame: sr={f.sample_rate} ch={f.num_channels} "
                  f"samples_per_channel={f.samples_per_channel} "
                  f"(SDK-resampled to {TARGET_SR} Hz mono; publisher's native rate is upstream)")
        samples = np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
        if f.num_channels > 1:
            samples = samples.reshape(-1, f.num_channels).mean(axis=1)
        buf.append(samples)
        total += len(samples)
        if total % (TARGET_SR * 2) < len(samples):
            print(f"  ...captured {total / TARGET_SR:.1f}s")
        if total >= CAPTURE_S * TARGET_SR:
            break
    await stream.aclose()
    done.set()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="demo")
    ap.add_argument("--identity", default="capture-test")
    args = ap.parse_args()

    token = make_token(args.identity, room=args.room, publish=False, subscribe=True)
    print(f"connecting to {URL} room={args.room!r} as {args.identity!r} (subscribe-only)")

    room = rtc.Room()
    buf: List[np.ndarray] = []
    info: dict = {}
    done = asyncio.Event()

    @room.on("track_subscribed")
    def on_track(track: rtc.Track, pub: rtc.RemoteTrackPublication,
                 participant: rtc.RemoteParticipant) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"subscribed to audio track from participant {participant.identity!r}")
            asyncio.create_task(_capture_track(track, buf, info, done))
        else:
            print(f"ignoring non-audio track from {participant.identity!r}")

    await room.connect(URL, token)
    print("connected. waiting for a remote audio track "
          "(publish from your phone via meet.livekit.io now)...")

    try:
        await asyncio.wait_for(done.wait(), timeout=120)
    except asyncio.TimeoutError:
        print("TIMED OUT after 120s.")
        if not buf:
            print("  -> never received audio. Is the phone publishing into the "
                  "SAME room name? Is its mic unmuted?")
    finally:
        await room.disconnect()

    if not buf:
        print("nothing captured - no test.wav written.")
        return

    audio = np.concatenate(buf)
    dur = len(audio) / TARGET_SR
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    with wave.open(OUT_PATH, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    print()
    print(f"captured {dur:.1f}s  frame_sr={info.get('sr', '?')} ch={info.get('ch', '?')}")
    print(f"wrote {OUT_PATH}  {dur:.1f}s  peak={peak:.3f}  rms={rms:.4f}")
    print()
    print("=== HOW TO READ THESE NUMBERS ===")
    print(" peak ~0.000            -> NO audio: phone mic muted/denied, wrong room,")
    print("                           or nothing was said. Fix before continuing.")
    print(" peak < 0.01            -> suspiciously quiet: either the /32768 int16")
    print("                           conversion is missing somewhere, or the mic")
    print("                           is nearly silent. Speak LOUDLY and re-run.")
    print(" peak 0.05-0.9, rms 0.01-0.15 -> healthy speech levels. Good to go.")
    print(" peak 1.000             -> clipping; back off from the mic a little.")
    print()
    print(" NOW ACTUALLY LISTEN TO test.wav (double-click it). Numbers can look")
    print(" fine while the audio is garbled - your ears are the real check.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\ninterrupted by Ctrl-C.")
