"""Klip kesme + 9:16 dikey kırpma.

İki mod:
- merkez kırpma (varsayılan): yükseklik üzerinden ortadan 9:16 kadraj.
- akıllı kırpma (--smart-crop): crop.motion_trajectory ile aksiyonu takip eden,
  zamanla yatay kayan pencere (tasarım Approach B).
Kaynak en-boy ne olursa olsun 1080x1920'a ölçeklenir.
"""
from __future__ import annotations

from pathlib import Path

from . import crop as cropper
from .ffmpeg_tools import FfmpegError, run

# 9:16 dikey, yükseklik üzerinden merkez kırpma -> 1080x1920.
# crop=ih*9/16:ih  => kaynağın tam yüksekliğini al, genişliği 9:16 oranına kırp.
VERTICAL_FILTER = (
    "crop='min(iw,ih*9/16)':ih,"
    "scale=1080:1920:force_original_aspect_ratio=decrease,"
    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
    "setsar=1"
)


def _smart_vertical_filter(x_expr: str) -> str:
    """Aksiyon-takipli x ifadesiyle dikey kırpma filtre zinciri.

    Parametre değerleri tek-tırnakla sarılır: ifadedeki virgüller literal kalır
    (ffmpeg filtergraph quoting), ayrıca x her kare için yeniden hesaplanır.
    """
    return (
        f"crop=w='min(iw,ih*9/16)':h='ih':x='{x_expr}':y='0',"
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


def cut_vertical(
    src: Path, start: float, duration: float, dst: Path, *, smart: bool = False,
) -> Path:
    """[start, start+duration] aralığını kesip 9:16 dikey klip üret.

    smart=True ise aksiyon-takipli kırpma denenir (crop.motion_trajectory).
    Hareket sinyali yoksa ya da örnekleme başarısızsa sessizce merkez kırpmaya
    düşülür — yani akıllı kırpma her zaman güvenli.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = VERTICAL_FILTER
    if smart:
        try:
            traj = cropper.motion_trajectory(src, start, duration)
            if traj:
                vf = _smart_vertical_filter(cropper.crop_x_expr(traj))
        except (FfmpegError, ValueError):
            vf = VERTICAL_FILTER       # güvenli geri düşüş: merkez kırpma
    run([
        "ffmpeg",
        "-v", "error",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "21",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ])
    return dst
