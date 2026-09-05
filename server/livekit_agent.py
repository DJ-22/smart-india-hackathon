"""LiveKit -> scoring-server bridge.

Joins a room subscribe-only, and PER PARTICIPANT IDENTITY keeps a separate
RingChunker and a separate session_id (the identity itself). Buffers are
never shared across tracks - two speakers produce two independent score
streams, not one interleaved mess.

The frame-receive loop never blocks on HTTP: completed windows go onto an
unbounded asyncio.Queue consumed by a background worker; if the worker
falls behind, the queue depth is logged LOUDLY instead of audio being
dropped silently.

Usage (from repo root):
  python server\\livekit_agent.py --room demo
  python server\\livekit_agent.py --room demo --identity laptop2
  python server\\livekit_agent.py --selftest        # no LiveKit needed

Ctrl-C drains the queue, disconnects, and prints windows-sent per
participant.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import wave as wave_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp
import numpy as np

from common.config import TARGET_SR
from server.chunker import RingChunker

log = logging.getLogger("server.livekit_agent")

BACKLOG_WARN_DEPTH = 3   # windows waiting; at 1 window/s/speaker this means seconds behind
DRAIN_TIMEOUT_S = 5.0


def to_wav_bytes(wav: np.ndarray, sr: int = TARGET_SR) -> bytes:
    buf = io.BytesIO()
    with wave_mod.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


@dataclass
class _ParticipantState:
    chunker: RingChunker = field(default_factory=RingChunker)
    windows_sent: int = 0
    windows_enqueued: int = 0
    first_score_logged: bool = False
    pump_task: Optional[asyncio.Task] = None


class ScoringAgent:
    def __init__(self, server_url: str, only_identity: Optional[str] = None) -> None:
        self.server_url = server_url.rstrip("/")
        self.only_identity = only_identity
        self.participants: Dict[str, _ParticipantState] = {}
        self.queue: asyncio.Queue = asyncio.Queue()  # unbounded: log depth, never drop
        self.http: Optional[aiohttp.ClientSession] = None
        self._worker: Optional[asyncio.Task] = None
        self._backlog_logged_at = 0.0

    # -- lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        self.http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10))
        self._worker = asyncio.create_task(self._score_worker())

    async def shutdown(self) -> None:
        for st in self.participants.values():
            if st.pump_task is not None:
                st.pump_task.cancel()
        depth = self.queue.qsize()
        if depth:
            log.info("draining %d queued windows (max %.0fs)...", depth, DRAIN_TIMEOUT_S)
        try:
            await asyncio.wait_for(self.queue.join(), timeout=DRAIN_TIMEOUT_S)
        except (asyncio.TimeoutError, TimeoutError):
            log.warning("drain timed out with %d windows unsent", self.queue.qsize())
        if self._worker is not None:
            self._worker.cancel()
        if self.http is not None:
            await self.http.close()

    def summary(self) -> str:
        if not self.participants:
            return "no participants were ever recorded."
        lines = ["windows sent per participant:"]
        for identity, st in sorted(self.participants.items()):
            lines.append(f"  {identity!r}: sent={st.windows_sent} "
                         f"enqueued={st.windows_enqueued}")
        return "\n".join(lines)

    # -- audio in ------------------------------------------------------------
    def want_track(self, identity: str) -> bool:
        if self.only_identity and identity != self.only_identity:
            log.info("skipping audio from %r (locked to %r)", identity, self.only_identity)
            return False
        st = self.participants.get(identity)
        if st is not None and st.pump_task is not None and not st.pump_task.done():
            log.info("already pumping audio for %r, ignoring extra track", identity)
            return False
        return True

    def on_samples(self, identity: str, samples: np.ndarray) -> None:
        """Feed decoded float32 mono 16k samples for ONE participant.
        Non-blocking: completed windows are queued for the HTTP worker."""
        st = self.participants.setdefault(identity, _ParticipantState())
        for window_id, window in st.chunker.push(samples):
            st.windows_enqueued += 1
            self.queue.put_nowait((identity, window_id, window))
        depth = self.queue.qsize()
        if depth >= BACKLOG_WARN_DEPTH:
            now = asyncio.get_running_loop().time()
            if now - self._backlog_logged_at >= 1.0:  # loud, but max 1 line/s
                self._backlog_logged_at = now
                log.warning("SCORE QUEUE BACKING UP: depth=%d (~%ds of audio waiting) "
                            "- is the scoring server slow or down?", depth, depth)

    async def pump_track(self, identity: str, track) -> None:
        from livekit import rtc

        st = self.participants.setdefault(identity, _ParticipantState())
        st.pump_task = asyncio.current_task()
        # Verified against livekit 1.1.17: these kwargs make the SDK
        # resample/downmix to 16 kHz mono for us.
        stream = rtc.AudioStream(track, sample_rate=TARGET_SR, num_channels=1)
        log.info("pumping audio for participant %r (session_id=%r)", identity, identity)
        try:
            async for ev in stream:
                f = ev.frame
                samples = np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
                if f.num_channels > 1:
                    samples = samples.reshape(-1, f.num_channels).mean(axis=1)
                self.on_samples(identity, samples)
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()
            log.info("audio stream for %r ended", identity)

    # -- scores out ----------------------------------------------------------
    async def _score_worker(self) -> None:
        assert self.http is not None
        while True:
            identity, window_id, window = await self.queue.get()
            try:
                async with self.http.post(
                    f"{self.server_url}/score",
                    data=to_wav_bytes(window),
                    headers={"X-Session-Id": identity,
                             "Content-Type": "application/octet-stream"},
                ) as resp:
                    if resp.status != 200:
                        log.warning("/score HTTP %d for %r window %d: %s",
                                    resp.status, identity, window_id,
                                    (await resp.text())[:200])
                        continue
                    body = await resp.json()
                st = self.participants[identity]
                st.windows_sent += 1
                if not st.first_score_logged:
                    st.first_score_logged = True
                    log.info("FIRST SCORE for session %r: %s", identity, body)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep serving
                log.warning("score POST failed for %r window %d: %s",
                            identity, window_id, exc)
            finally:
                self.queue.task_done()


# --------------------------------------------------------------------------
# LiveKit wiring
# --------------------------------------------------------------------------
async def run_agent(args: argparse.Namespace) -> None:
    from livekit import rtc

    from server.tokens import URL, make_token

    agent = ScoringAgent(args.server, args.identity)
    await agent.start()

    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track(track: "rtc.Track", pub: "rtc.RemoteTrackPublication",
                 participant: "rtc.RemoteParticipant") -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        identity = participant.identity
        if not agent.want_track(identity):
            return
        asyncio.create_task(agent.pump_track(identity, track))

    token = make_token(args.join_as, room=args.room, publish=False, subscribe=True)
    await room.connect(URL, token)
    log.info("connected to %s room=%r as %r (subscribe-only); scoring via %s",
             URL, args.room, args.join_as, args.server)
    if args.identity:
        log.info("locked to participant %r", args.identity)

    try:
        await asyncio.Event().wait()  # run until Ctrl-C
    except asyncio.CancelledError:
        pass
    finally:
        log.info("shutting down...")
        await agent.shutdown()
        await room.disconnect()
        print(agent.summary())


# --------------------------------------------------------------------------
# self-test: full agent pipeline minus LiveKit, against a running server
# --------------------------------------------------------------------------
async def run_selftest(args: argparse.Namespace) -> None:
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "win3s.wav"
    with wave_mod.open(str(fixture)) as w:
        clip = np.frombuffer(w.readframes(w.getnframes()),
                             dtype=np.int16).astype(np.float32) / 32768.0
    audio = np.tile(clip, 4)  # 12 s of speech

    agent = ScoringAgent(args.server, args.identity)
    await agent.start()
    try:
        # two fake participants, frames interleaved 10 ms at a time, to
        # prove per-participant isolation end to end
        frame = 160
        for i in range(0, len(audio), frame):
            for identity in ("selftest-a", "selftest-b"):
                if agent.only_identity and identity != agent.only_identity:
                    continue
                agent.on_samples(identity, audio[i : i + frame])
            await asyncio.sleep(0)  # let the worker run, as live frames would
        await asyncio.wait_for(agent.queue.join(), timeout=30)
    finally:
        await agent.shutdown()
    print(agent.summary())
    from common.config import HOP_S, WINDOW_S

    expected = 1 + (len(audio) - int(WINDOW_S * TARGET_SR)) // int(HOP_S * TARGET_SR)
    for identity, st in agent.participants.items():
        assert st.windows_sent == expected, (identity, st.windows_sent, expected)
    print(f"selftest PASS: {expected} windows per participant, all scored")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--room", default="demo")
    ap.add_argument("--server", default="http://127.0.0.1:8000",
                    help="scoring server base URL")
    ap.add_argument("--identity", default=None,
                    help="lock to ONE participant; default: score every speaker separately")
    ap.add_argument("--join-as", default="detector-agent",
                    help="identity the agent joins the room as")
    ap.add_argument("--selftest", action="store_true",
                    help="no LiveKit: pump fixture audio for two fake participants "
                         "through the full chunk->queue->POST pipeline")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run_selftest(args) if args.selftest else run_agent(args))
    except KeyboardInterrupt:
        print("\ninterrupted by Ctrl-C.")


if __name__ == "__main__":
    main()
