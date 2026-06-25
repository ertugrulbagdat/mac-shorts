"""Altyazı: faster-whisper ile transkript -> SRT -> videoya gömme.

Tasarım v1: sade altyazı. faster-whisper modeli ilk çalıştırmada inilir.
Model inmezse ya da gömme başarısızsa, hat çökmesin: yanına .srt bırakılır
ve uyarı verilir (graceful degrade).
"""
from __future__ import annotations

from pathlib import Path

from .ffmpeg_tools import FfmpegError, run

_MODEL_CACHE: dict[str, object] = {}


def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_to_srt(clip: Path, srt_path: Path, model_name: str, lang: str | None) -> bool:
    """Klibi transkript edip SRT yaz. Başarılıysa True."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  ! faster-whisper kurulu değil, altyazı atlandı "
              "(`pip install faster-whisper`).")
        return False

    try:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            _MODEL_CACHE[model_name] = model
        segments, _info = model.transcribe(str(clip), language=lang, vad_filter=True)
    except Exception as e:
        print(f"  ! Transkript başarısız, altyazı atlandı: {e}")
        return False

    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}")
        lines.append(text)
        lines.append("")
    if not lines:
        return False
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def _escape_for_filter(p: Path) -> str:
    """Windows yolunu ffmpeg subtitles filtresi için kaçışla."""
    s = str(p.resolve()).replace("\\", "/")
    # sürücü harfindeki ikinci nokta filtre ayracıyla karışmasın
    s = s.replace(":", "\\:")
    return s


def burn(
    clip: Path,
    srt_path: Path,
    dst: Path,
    *,
    font_size: int = 12,
    margin_v: int = 45,
) -> bool:
    """SRT'yi videoya göm. libass yoksa/başarısızsa False döner.

    Değerler libass'ın varsayılan SRT tuvalinde (yaklaşık 288 yükseklik)
    yorumlanır; gerçek 1080x1920 videoya ölçeklenir. Yani font_size küçük
    sayılardır (varsayılan 12). margin_v küçüldükçe altyazı AŞAĞI iner
    (varsayılan 45; eski 120 ekranın ortasıydı).
    """
    style = (
        f"FontName=Arial,FontSize={font_size},PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
        f"Alignment=2,MarginV={margin_v}"
    )
    vf = f"subtitles='{_escape_for_filter(srt_path)}':force_style='{style}'"
    try:
        run([
            "ffmpeg",
            "-v", "error",
            "-y",
            "-i", str(clip),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "21",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(dst),
        ])
        return True
    except FfmpegError as e:
        print(f"  ! Altyazı gömme başarısız (SRT yan dosya olarak bırakıldı): "
              f"{str(e)[:200]}")
        return False
