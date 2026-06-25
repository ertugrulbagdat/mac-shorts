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
    min_len: float = 4.0,
    max_len: float = 28.0,
    scene_threshold: float = 0.35,
) -> list[Moment]:
    """Özet videodan en enerjik N parçayı seç.

    Mantık: sahne kesimleri videoyu doğal parçalara (atak/gol) böler;
    her parçanın ortalama ses enerjisi onu sıralar; en yüksek N parça seçilir.
    """
    duration = media_duration(src)
    cuts = scene_cuts(src, scene_threshold)
    boundaries = [0.0, *[c for c in cuts if 0 < c < duration], duration]
    boundaries = sorted(set(boundaries))

    samples, sr = extract_pcm(src)
    env, win_s = _rms_envelope(samples, sr)

    def energy_between(a: float, b: float) -> tuple[float, float]:
        i0 = int(a / win_s)
        i1 = max(i0 + 1, int(b / win_s))
        seg = env[i0:i1]
        if len(seg) == 0:
            return 0.0, a
        peak_idx = i0 + int(np.argmax(seg))
        return float(np.mean(seg)), peak_idx * win_s

    segments: list[Moment] = []
    for a, b in zip(boundaries, boundaries[1:]):
        seg_len = b - a
        if seg_len < min_len:
            continue
        score, peak = energy_between(a, b)
        # Çok uzun parçayı zirve etrafında kırp.
        if seg_len > max_len:
            start = max(a, peak - max_len * 0.35)
            end = min(b, start + max_len)
        else:
            start, end = a, b
        segments.append(Moment(start=start, end=end, peak=peak, score=score))

    # Sahne kesimi hiç çıkmadıysa (tek parça video) ses zirvelerine düş.
    if len(segments) <= 1:
        return _peaks_fallback(env, win_s, duration, count, min_len, max_len)

    segments.sort(key=lambda m: m.score, reverse=True)
    chosen = _dedupe(segments, count)
    chosen.sort(key=lambda m: m.start)
    return chosen


def _peaks_fallback(
    env: np.ndarray, win_s: float, duration: float, count: int,
    min_len: float, max_len: float,
) -> list[Moment]:
    """Sahne kesimi yoksa: en yüksek enerji zirvelerini al, etrafından kes."""
    clip_len = min(max_len, max(min_len, 18.0))
    order = np.argsort(env)[::-1]
    chosen: list[Moment] = []
    used: list[float] = []
    for idx in order:
        peak = idx * win_s
        if any(abs(peak - u) < clip_len for u in used):
            continue
        start = max(0.0, peak - 6.0)
        end = min(duration, start + clip_len)
        chosen.append(Moment(start=start, end=end, peak=peak, score=float(env[idx])))
        used.append(peak)
        if len(chosen) >= count:
            break
    chosen.sort(key=lambda m: m.start)
    return chosen


def _dedupe(segments: list[Moment], count: int) -> list[Moment]:
    """Birbirine çok yakın (örtüşen) anları ele, en iyi N'i döndür."""
    chosen: list[Moment] = []
    for m in segments:
        if any(not (m.end <= c.start or m.start >= c.end) for c in chosen):
            continue  # örtüşüyor
        chosen.append(m)
        if len(chosen) >= count:
            break
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
