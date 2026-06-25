"""Heyecanlı an tespiti.

İki mod (tasarım dokümanı, Recommended Approach):
- highlights: videonun kendi sahne kesimleri + ses enerjisi sıralaması.
- match (tam maç): elle girilen gol dakikaları + pencere içi ses zirvesiyle
  saniyeye hizalama.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg_tools import FfmpegError, extract_pcm, ffmpeg_path, media_duration


@dataclass
class Moment:
    """Bir aday klip anı."""
    start: float          # klip başlangıcı (sn)
    end: float            # klip bitişi (sn)
    peak: float           # heyecanın doruk saniyesi (sn)
    score: float          # göreli enerji skoru (sıralama için)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _rms_envelope(samples: np.ndarray, sr: int, win_s: float = 0.5) -> tuple[np.ndarray, float]:
    """Pencere başına RMS enerji zarfı. Returns: (zarf, pencere_süresi)."""
    win = max(1, int(sr * win_s))
    n = len(samples) // win
    if n == 0:
        return np.array([float(np.sqrt(np.mean(samples**2)) if len(samples) else 0.0)]), win_s
    trimmed = samples[: n * win].reshape(n, win)
    env = np.sqrt(np.mean(trimmed**2, axis=1))
    return env, win_s


def scene_cuts(src: Path, threshold: float = 0.35) -> list[float]:
    """ffmpeg sahne tespitiyle sahne değişim zaman damgaları (sn)."""
    proc = subprocess.run(
        [
            ffmpeg_path(),
            "-v", "info",
            "-i", str(src),
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # showinfo satırları: "... pts_time:12.345 ..."
    times = [float(t) for t in re.findall(r"pts_time:(\d+(?:\.\d+)?)", proc.stderr or "")]
    return sorted(set(times))


def detect_highlights(
    src: Path,
    count: int,
    *,
    min_len: float = 6.0,
    pre: float = 8.0,
    post: float = 12.0,
    skip_intro: float = 12.0,
    skip_outro: float = 8.0,
    baseline_s: float = 20.0,
    scene_threshold: float = 0.35,   # geriye dönük uyum için tutulur; kullanılmaz
) -> list[Moment]:
    """Özet videodan gol/heyecan anlarını seç.

    Gol imzası = ANİ ses zirvesi (spiker bağırması + tribün uğultusu). Düz intro
    müziği ortalama enerjide yüksek ama "ani" değildir. Bu yüzden mutlak enerji
    yerine YEREL ZEMİNE göre çıkıntı (prominence) kullanılır: golün ani çıkışı
    hareketli ortalamanın üstüne fırlar, sabit intro müziği fırlamaz.

    Ayrıca intro (ilk skip_intro sn) ve outro (son skip_outro sn) atlanır; her
    zirvenin etrafından gol + kutlama için bir pencere alınır (pre/post).
    """
    duration = media_duration(src)
    samples, sr = extract_pcm(src)
    env, win_s = _rms_envelope(samples, sr, win_s=0.5)
    if len(env) == 0:
        return []

    # Yerel zemin (hareketli ortalama). Golün ani sıçraması bunun üstüne çıkar.
    base_win = max(1, int(baseline_s / win_s))
    kernel = np.ones(base_win) / base_win
    baseline = np.convolve(env, kernel, mode="same")
    prominence = np.clip(env - baseline, 0.0, None)

    min_gap = pre + post              # seçilen anlar örtüşmesin / aynı gol tekrar gelmesin
    order = np.argsort(prominence)[::-1]
    chosen: list[Moment] = []
    used: list[float] = []
    for idx in order:
        t = float(idx) * win_s
        if t < skip_intro or t > duration - skip_outro:
            continue                  # intro/outro'yu atla
        if any(abs(t - u) < min_gap for u in used):
            continue                  # önceki bir anla çakışıyor
        start = max(0.0, t - pre)
        end = min(duration, t + post)
        if end - start < min_len:
            continue
        chosen.append(Moment(start=start, end=end, peak=t, score=float(prominence[idx])))
        used.append(t)
        if len(chosen) >= count:
            break

    chosen.sort(key=lambda m: m.start)
    return chosen


def parse_minutes(spec: str) -> list[float]:
    """"23,45+2,67" -> [1380.0, 2820.0, 4020.0] (saniye)."""
    out: list[float] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "+" in part:
            base, extra = part.split("+", 1)
            minute = int(base) + int(extra)
        else:
            minute = int(part)
        out.append(minute * 60.0)
    return out


def detect_match(
    src: Path,
    minutes_spec: str,
    *,
    search_window: float = 45.0,
    pre: float = 6.0,
    post: float = 10.0,
) -> list[Moment]:
    """Tam maç: elle girilen dakikaların etrafında ses zirvesiyle saniyeye hizala."""
    duration = media_duration(src)
    centers = parse_minutes(minutes_spec)
    if not centers:
        raise FfmpegError("Tam maç modu için --minutes gerekli, örn: 23,45+2,67")

    samples, sr = extract_pcm(src)
    env, win_s = _rms_envelope(samples, sr, win_s=0.25)

    moments: list[Moment] = []
    for c in centers:
        c = min(c, duration - 1)
        lo = max(0.0, c - search_window)
        hi = min(duration, c + search_window)
        i0, i1 = int(lo / win_s), max(int(lo / win_s) + 1, int(hi / win_s))
        seg = env[i0:i1]
        if len(seg) == 0:
            peak = c
            score = 0.0
        else:
            peak = (i0 + int(np.argmax(seg))) * win_s
            score = float(np.max(seg))
        start = max(0.0, peak - pre)
        end = min(duration, peak + post)
        moments.append(Moment(start=start, end=end, peak=peak, score=score))
    return moments
