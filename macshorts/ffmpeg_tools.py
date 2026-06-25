"""ffmpeg/ffprobe bulma ve düşük seviye yardımcılar.

Sistemde ffmpeg yoksa imageio-ffmpeg'in gömülü binary'sine düşer.
ffprobe'a bağımlı değiliz: süre, sesi çözerek (PCM örnek sayısı / örnekleme
hızı) hesaplanır.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


class FfmpegError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    """Sistem ffmpeg'ini bul, yoksa imageio-ffmpeg gömülü binary'sini kullan."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - kurulum eksikse
        raise FfmpegError(
            "ffmpeg bulunamadı. Sistemde ffmpeg kurun ya da "
            "`pip install imageio-ffmpeg` çalıştırın."
        ) from e


@functools.lru_cache(maxsize=1)
def ytdlp_ffmpeg_dir() -> str:
    """yt-dlp'nin tanıyabileceği şekilde 'ffmpeg(.exe)' içeren bir klasör döndür.

    yt-dlp ffmpeg binary'sini adıyla ('ffmpeg') arar. imageio-ffmpeg'in binary
    adı farklı olduğundan, bir kez doğru adla önbelleğe kopyalanır ve o klasör
    verilir. Sistem ffmpeg'i varsa onun klasörü kullanılır.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return str(Path(exe).parent)
    src = Path(ffmpeg_path())
    cache = Path.home() / ".cache" / "macshorts" / "bin"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists() or target.stat().st_size != src.stat().st_size:
        shutil.copy2(src, target)
    return str(cache)


def run(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    """ffmpeg'i verilen argümanlarla çalıştır. args[0] 'ffmpeg' olmalı."""
    if args and args[0] == "ffmpeg":
        args = [ffmpeg_path(), *args[1:]]
    proc = subprocess.run(
        args,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1500:]
        raise FfmpegError(
            f"ffmpeg başarısız (kod {proc.returncode}):\n{tail}"
        )
    return proc


def extract_pcm(src: Path, sample_rate: int = 1000) -> tuple[np.ndarray, int]:
    """Sesi mono float32 dizisine çöz. Enerji/zirve analizi için kullanılır.

    sample_rate küçük tutulur (varsayılan 1000 Hz) çünkü 90 dakikalık maçı bile
    birkaç MB'a indirir ve ms çözünürlüğü heyecanlı an tespitine yeter.

    Returns: (örnekler, örnekleme_hızı)
    """
    args = [
        "ffmpeg",
        "-v", "error",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-",
    ]
    proc = subprocess.run(
        [ffmpeg_path(), *args[1:]],
        capture_output=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-1500:]
        raise FfmpegError(f"Ses çözülemedi:\n{tail}")
    raw = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return raw, sample_rate


def media_duration(src: Path) -> float:
    """Saniye cinsinden süre. ffprobe yerine showinfo'dan son kare zamanını okur."""
    proc = subprocess.run(
        [
            ffmpeg_path(),
            "-v", "error",
            "-i", str(src),
            "-f", "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # ffmpeg null muxer stderr'e "time=HH:MM:SS.xx" yazar; en sonuncusunu al.
    import re

    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
    if not times:
        # yedek: sesten süre
        samples, sr = extract_pcm(src)
        return len(samples) / sr
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)
