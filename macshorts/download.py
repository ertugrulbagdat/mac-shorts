"""YouTube indirme (yt-dlp) ve yerel dosya geçişi.

NOT (tasarım dokümanı, Open Questions): Telifli maç içeriğini indirmek
YouTube ToS'una aykırı olabilir. Bu araç kişisel/deneysel kullanım içindir;
ne indirip yayınladığının sorumluluğu kullanıcıdadır.
"""
from __future__ import annotations

from pathlib import Path

from .ffmpeg_tools import ytdlp_ffmpeg_dir


def fetch(url_or_path: str, out_dir: Path) -> tuple[Path, dict]:
    """URL ise indir, yerel dosya ise olduğu gibi döndür.

    Returns: (video_yolu, kaynak_metadata). Kaynak metadata indirilen videonun
    kendi başlığı/caption'ı/yükleyeni; başlık/açıklama üretiminde kullanılır.
    Yerel dosyada metadata boştur.
    """
    p = Path(url_or_path)
    if p.exists():
        return p, {}
    return _download(url_or_path, out_dir)


def _extract_meta(info: dict) -> dict:
    """yt-dlp info dict'inden anlamlı kaynak metadata'sı çıkar."""
    return {
        "title": info.get("title") or "",
        "caption": info.get("description") or "",  # Instagram'da caption burada
        "uploader": info.get("uploader") or info.get("channel") or info.get("uploader_id") or "",
        "tags": info.get("tags") or info.get("categories") or [],
        "url": info.get("webpage_url") or "",
    }


def _download(url: str, out_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp kurulu değil. `pip install yt-dlp` çalıştırın."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "source-%(id)s.%(ext)s")
    ffmpeg_dir = ytdlp_ffmpeg_dir()

    captured: dict[str, str] = {}

    def hook(d: dict) -> None:
        if d.get("status") == "finished":
            captured["path"] = d.get("filename", "")

    opts = {
        # 9:16 kırpma için en az 1080 yükseklik yeterli; aşırı 4K indirmeyi önle.
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
        "outtmpl": template,
        "merge_output_format": "mp4",
        "ffmpeg_location": ffmpeg_dir,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    meta = _extract_meta(info)

    # Birleştirme sonrası gerçek dosyayı bul.
    if captured.get("path") and Path(captured["path"]).exists():
        final = Path(captured["path"])
        # merge sonrası uzantı mp4 olabilir
        if final.suffix != ".mp4":
            mp4 = final.with_suffix(".mp4")
            if mp4.exists():
                return mp4, meta
        return final, meta

    vid = info.get("id", "")
    matches = sorted(out_dir.glob(f"source-{vid}.*"))
    if not matches:
        raise RuntimeError("İndirme tamamlandı ama dosya bulunamadı.")
    # mp4'ü tercih et
    for m in matches:
        if m.suffix == ".mp4":
            return m, meta
    return matches[0], meta
