"""Cache IndicTTS-Deepfake clips locally as 16 kHz mono WAV + manifest.csv.

The HF parquet export is ~18 GB (44.1 kHz source audio), far too large to pull
during a sprint. We instead use the HF datasets-server row API, which hands out
per-row signed WAV URLs, so we can download a stratified subset only.

Two phases, both resumable:
  python ml/fetch_data.py harvest    -> data/index.jsonl  (metadata + urls)
  python ml/fetch_data.py download N -> data/clips/*.wav + data/manifest.csv

Download order is a round-robin over (language, label) so that *any* prefix of
the downloaded set stays balanced -- kill it whenever and the data is usable.
"""
import csv
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import CLIPS_DIR, DATA_DIR, HF_DATASET, INDEX_JSONL, MANIFEST, SR

BASE = "https://datasets-server.huggingface.co"
PAGE = 100
MAX_KEEP_S = 10.0          # clips get cropped to 4 s at train time anyway
N_DEFAULT = 6000
WORKERS = 20


def _get(path, params, tries=8, timeout=180):
    """GET with backoff. The datasets-server 429s readily; be patient, not parallel."""
    last = None
    for i in range(tries):
        try:
            r = requests.get(BASE + "/" + path, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = "HTTP %s" % r.status_code
            if r.status_code == 429:
                time.sleep(min(60, 5 * (i + 1)))
                continue
        except Exception as e:                                   # noqa: BLE001
            last = repr(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError("%s failed after %d tries: %s" % (path, tries, last))


def n_rows(split="train"):
    j = _get("rows", {"dataset": HF_DATASET, "config": "default",
                      "split": split, "offset": 0, "length": 1})
    return j["num_rows_total"]


def harvest(split="train", refresh=False):
    """Page the row API and write one JSON line per clip (metadata + signed url).

    The signed asset URLs expire after roughly a day. Re-run with refresh=True
    (`python ml/fetch_data.py refresh`) to append fresh ones before a download;
    load_index() takes the newest entry for each row.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    total = n_rows(split)
    print("%s: %d rows%s" % (split, total, " (refreshing urls)" if refresh else ""),
          flush=True)

    done = set()
    if os.path.exists(INDEX_JSONL) and not refresh:
        with open(INDEX_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["row_idx"])
                except Exception:                                # noqa: BLE001
                    pass
        print("resuming, %d rows already indexed" % len(done), flush=True)

    offsets = [o for o in range(0, total, PAGE)
               if not all(i in done for i in range(o, min(o + PAGE, total)))]
    lock = threading.Lock()
    fh = open(INDEX_JSONL, "a", encoding="utf-8")

    def page(off):
        j = _get("rows", {"dataset": HF_DATASET, "config": "default",
                          "split": split, "offset": off, "length": PAGE})
        out = []
        for row in j["rows"]:
            r = row["row"]
            out.append({
                "row_idx": row["row_idx"],
                "id": r["id"],
                "language": r["language"],
                "label": int(r["is_tts"]),      # 1 = fake/TTS, 0 = real
                "url": r["audio"][0]["src"],
            })
        with lock:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        return len(out)

    got, bad = 0, []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(page, o): o for o in offsets}
        for k, f in enumerate(futs):
            try:
                got += f.result()
            except Exception as e:                               # noqa: BLE001
                bad.append(futs[f])
                if len(bad) <= 3:
                    print("  page %s failed: %r" % (futs[f], e), flush=True)
            if k % 10 == 0:
                print("  indexed %d (%d failed pages)" % (got, len(bad)), flush=True)
    fh.close()
    print("harvest: +%d rows, %d failed pages -> %s" % (got, len(bad), INDEX_JSONL),
          flush=True)
    return len(bad)


def load_index():
    """Last entry per row wins -- a re-harvest appends fresher signed URLs, and
    the asset URLs expire after about a day, so preferring the first copy hands
    you a file of guaranteed-403 links."""
    by_row = {}
    with open(INDEX_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_row[r["row_idx"]] = r
    return [by_row[k] for k in sorted(by_row)]


def speaker_of(clip_id, language):
    """Pseudo-speaker from the clip id. The corpus mixes three id conventions:

        ASM_F_ANGER_00342          -> ASM_F      (most IndicTTS languages)
        te_m_00123                 -> te_m       (English, Hindi, Telugu)
        train_manipurifemale_07321 -> Manipuri_female
        text1517                   -> <language>_unk   (Odia: no speaker in the id)

    There are at most two speakers per language, so this is a pseudo-speaker,
    not a real speaker id -- see RESULTS.md for what that costs us.
    """
    parts = clip_id.split("_")
    if len(parts) >= 3 and parts[0] in ("train", "test", "val"):
        tag = parts[1].lower()
        for g in ("female", "male"):
            if tag.endswith(g):
                return "%s_%s" % (language, g)
        return "%s_%s" % (language, tag)
    if len(parts) >= 2 and len(parts[1]) <= 2:
        return parts[0] + "_" + parts[1]
    return language + "_unk"


def _balanced_order(rows):
    """Round-robin over (language, label) so any prefix stays balanced."""
    buckets = {}
    for r in rows:
        buckets.setdefault((r["language"], r["label"]), []).append(r)
    rng = np.random.default_rng(42)
    for v in buckets.values():
        rng.shuffle(v)
    keys = sorted(buckets)
    out, i = [], 0
    while any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        i += 1
    return out


def download(n=N_DEFAULT):
    os.makedirs(CLIPS_DIR, exist_ok=True)
    order = _balanced_order(load_index())[:n]
    print("downloading %d clips with %d workers" % (len(order), WORKERS), flush=True)

    lock = threading.Lock()
    state = {"ok": 0, "skip": 0, "fail": 0, "t0": time.time(), "bytes": 0}
    rows_out = []

    def one(r):
        # the id is dataset-supplied; strip anything that could escape CLIPS_DIR
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(r["id"]))[:80]
        path = os.path.join(CLIPS_DIR, "%05d_%s.wav" % (int(r["row_idx"]), safe_id))
        if os.path.exists(path) and os.path.getsize(path) > 4096:
            try:
                dur = sf.info(path).duration
                with lock:
                    state["skip"] += 1
                    rows_out.append(dict(r, path=path, dur_s=round(dur, 2)))
                return
            except Exception:                                    # noqa: BLE001
                pass
        try:
            raw = None
            for attempt in range(5):
                resp = requests.get(r["url"], timeout=180)
                if resp.status_code == 200:
                    raw = resp.content
                    break
                time.sleep(3 * (attempt + 1))       # asset CDN 429s under load
            if raw is None:
                raise RuntimeError("HTTP %s" % resp.status_code)
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR:
                g = int(np.gcd(int(sr), SR))
                wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
            wav = wav[: int(MAX_KEEP_S * SR)]
            peak = float(np.max(np.abs(wav))) if wav.size else 0.0
            if wav.size < int(0.4 * SR) or peak < 1e-4:
                with lock:
                    state["fail"] += 1
                return
            sf.write(path, wav, SR, subtype="PCM_16")
            with lock:
                state["ok"] += 1
                state["bytes"] += len(raw)
                rows_out.append(dict(r, path=path, dur_s=round(len(wav) / SR, 2)))
        except Exception as e:                                   # noqa: BLE001
            with lock:
                state["fail"] += 1
                if state["fail"] <= 5:
                    print("  fail %s: %r" % (r["id"], e), flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, r) for r in order]
        last = 0
        while any(not f.done() for f in futs):
            time.sleep(5)
            n_done = state["ok"] + state["skip"] + state["fail"]
            if n_done - last >= 100:
                last = n_done
                el = time.time() - state["t0"]
                mb = state["bytes"] / 1e6
                print("  %d/%d ok=%d skip=%d fail=%d  %.0fMB  %.1fMB/s  %.0fs"
                      % (n_done, len(order), state["ok"], state["skip"],
                         state["fail"], mb, mb / max(el, 1), el), flush=True)
        for f in futs:
            f.result()

    write_manifest(rows_out)
    print("done in %.0fs  ok=%d skip=%d fail=%d"
          % (time.time() - state["t0"], state["ok"], state["skip"], state["fail"]),
          flush=True)


def write_manifest(rows_out):
    rows_out.sort(key=lambda r: r["row_idx"])
    fields = ["path", "label", "language", "speaker", "id", "dur_s", "row_idx"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow({
                "path": os.path.relpath(r["path"], DATA_DIR).replace("\\", "/"),
                "label": r["label"],
                "language": r["language"],
                "speaker": speaker_of(r["id"], r["language"]),
                "id": r["id"],
                "dur_s": r["dur_s"],
                "row_idx": r["row_idx"],
            })
    print("manifest -> %s  (%d rows)" % (MANIFEST, len(rows_out)), flush=True)


def rebuild_manifest():
    """Rebuild manifest.csv from whatever WAVs are already on disk."""
    idx = {r["row_idx"]: r for r in load_index()}
    rows_out = []
    for fn in os.listdir(CLIPS_DIR):
        if not fn.endswith(".wav"):
            continue
        r = idx.get(int(fn.split("_")[0]))
        if r is None:
            continue
        p = os.path.join(CLIPS_DIR, fn)
        try:
            dur = sf.info(p).duration
        except Exception:                                        # noqa: BLE001
            continue
        rows_out.append(dict(r, path=p, dur_s=round(dur, 2)))
    write_manifest(rows_out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "rebuild":
        rebuild_manifest()
    else:
        if cmd in ("harvest", "refresh", "all"):
            for attempt in range(6):          # resumable; keep going until complete
                if harvest("train", refresh=(cmd == "refresh")) == 0:
                    break
                print("retrying failed pages (attempt %d)" % (attempt + 2), flush=True)
                time.sleep(20)
        if cmd in ("download", "all"):
            download(int(sys.argv[2]) if len(sys.argv) > 2 else N_DEFAULT)
