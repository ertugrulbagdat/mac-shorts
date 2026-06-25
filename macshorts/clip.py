"""Klip kesme + 9:16 dikey kırpma.

v1: merkez kırpma (akıllı/saliency kırpma tasarımda B'ye ertelendi).
Kaynak en-boy ne olursa olsun, yükseklik üzerinden 9:16 dikey kadraj alınır
ve 1080x1920'a ölçeklenir.
"""
from __future__ import annotations

from pathlib import Path

from .ffmpeg_tools import run

# 9:16 dikey, yükseklik üzerinden merkez kırpma -> 1080x1920.
# crop=ih*9/16:ih  => kaynağın tam yüksekliğini al, genişliği 9:16 oranına kırp.
VERTICAL_FILTER = (
    "crop='min(iw,ih*9/16)':ih,"
    "scale=1080:1920:force_original_aspect_ratio=decrease,"
    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
    "setsar=1"
)


def cut_segment(src: Path, start: float, duration: float, dst: Path) -> Path:
    """[start, start+duration] aralığını en-boyu KORUYARAK kes (9:16'ya çevirmez)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg",
        "-v", "error",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "21",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ])
    return dst


def cut_vertical(src: Path, start: float, duration: float, dst: Path) -> Path:
    """[start, start+duration] aralığını kesip 9:16 dikey klip üret."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg",
        "-v", "error",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-vf", VERTICAL_FILTER,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "21",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ])
    return dst
