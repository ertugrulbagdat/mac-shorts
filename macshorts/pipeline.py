"""Üretim hattı orkestrasyonu.

Akış (tasarım dokümanı, Recommended Approach):
  1. Girdi: URL/dosya + mod (highlights / match)
  2. yt-dlp ile indir (URL ise)
  3. Aday anları bul (özet: sahne+ses, tam maç: elle dakika + ses zirvesi)
  4. Her aday için 9:16 klip kes
  5. faster-whisper ile altyazı (varsayılan açık)
  6. Manifest yaz; YAYINDAN ÖNCE ZORUNLU İNSAN KONTROLÜ
Yayın YOK: araç sadece klip üretir, paylaşma kararı insanda.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import clip as clipper
from . import detect, download, subtitles
from .detect import Moment
from .ffmpeg_tools import media_duration


@dataclass
class ClipResult:
    index: int
    file: str
    start: float
    end: float
    peak: float
    duration: float
    score: float
    subtitled: bool
    srt: str | None
    suggested_title: str
    youtube_url: str | None = None


@dataclass
class Options:
    source: str
    mode: str = "highlights"          # highlights | match | whole
    count: int = 5
    vertical: bool = False            # whole modunda 9:16'ya zorla (varsayılan: orijinal en-boy)
    short_len: float = 60.0           # whole modunda bu süreden uzun video parçalanır (sn)
    minutes: str | None = None        # match modu için "23,45+2"
    out_dir: Path = Path("output")
    subtitles: bool = True
    whisper_model: str = "small"
    lang: str | None = None
    scene_threshold: float = 0.35
    label: str = "klip"               # önerilen başlık öneki
    sub_size: int = 12                # altyazı font boyutu (libass SRT tuvali); küçük sayı
    sub_margin: int = 45              # altyazı alt boşluğu; küçüldükçe daha aşağı
    publish: bool = False             # YouTube'a yarı-otomatik yükleme
    privacy: str = "private"          # private | unlisted | public (varsayılan private)
    client_secret: Path = Path("client_secret.json")
    token_path: Path = Path("youtube_token.json")


def run(opts: Options) -> list[ClipResult]:
    out_dir = Path(opts.out_dir)
    work = out_dir / datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    work.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Kaynak alınıyor: {opts.source}")
    src = download.fetch(opts.source, work / "_source")
    print(f"      -> {src}")

    if opts.mode == "whole":
        return _process_whole(src, work, opts)

    print(f"[2/4] Anlar tespit ediliyor (mod={opts.mode}) ...")
    moments = _detect(src, opts)
    if not moments:
        print("      ! Hiç aday an bulunamadı.")
        return []
    print(f"      -> {len(moments)} aday an")

    print(f"[3/4] Klipler kesiliyor (9:16){' + altyazı' if opts.subtitles else ''} ...")
    results: list[ClipResult] = []
    for i, m in enumerate(moments, start=1):
        res = _make_clip(src, m, i, work, opts)
        results.append(res)
        print(f"      [{i}/{len(moments)}] {res.file} "
              f"({res.duration:.1f}sn{', altyazılı' if res.subtitled else ''})")

    _maybe_publish(results, opts)

    print("[4/4] Manifest yazılıyor ...")
    _write_manifest(work, src, opts, results)
    print(f"\nBitti. {len(results)} klip: {work}")
    print("UYARI: Yayınlamadan ÖNCE klipleri elle izle ve telif riskini kabul "
          "ettiğini doğrula. Araç otomatik yayın YAPMAZ.")
    return results


def _whole_segments(dur: float, short_len: float) -> list[tuple[float, float]]:
    """Videoyu Shorts'a uygun parçalara böl.

    dur <= short_len ise tek parça (tüm video). Aksi halde ardışık short_len'lik
    parçalar; son parça kalan kadar. Çok kısa (< 3sn) son parça öncekine eklenir.
    """
    if dur <= short_len:
        return [(0.0, dur)]
    segs: list[tuple[float, float]] = []
    t = 0.0
    while t < dur - 0.1:
        seg = min(short_len, dur - t)
        segs.append((t, seg))
        t += seg
    if len(segs) >= 2 and segs[-1][1] < 3.0:
        s0, d0 = segs[-2]
        _, d1 = segs.pop()
        segs[-1] = (s0, d0 + d1)
    return segs


def _process_whole(src: Path, work: Path, opts: Options) -> list[ClipResult]:
    """whole modu: videoyu indir + altyazı ekle.

    Video <= short_len (varsayılan 60sn) ise tek parça (çerçeveye dokunulmaz).
    Daha uzunsa Shorts sınırına uyacak şekilde ardışık parçalara bölünür.
    --vertical verilirse parça(lar) 9:16'ya kırpılır.
    """
    dur = media_duration(src)
    segments = _whole_segments(dur, opts.short_len)
    multi = len(segments) > 1

    if multi:
        print(f"[2/3] Video {dur:.0f}sn > {opts.short_len:.0f}sn -> "
              f"{len(segments)} parçaya bölünüyor ...")
    else:
        print("[2/3] Video tek parça (≤ sınır), parçalanmıyor ...")
    print(f"[3/3] Parça(lar) işleniyor"
          f"{' + altyazı' if opts.subtitles else ''} ...")

    results: list[ClipResult] = []
    for i, (start, seg_dur) in enumerate(segments, start=1):
        res = _whole_segment_clip(
            src, start, seg_dur, i, work, opts, single=not multi, total_dur=dur,
        )
        results.append(res)
        print(f"      [{i}/{len(segments)}] {Path(res.file).name} "
              f"({res.duration:.1f}sn{', altyazılı' if res.subtitled else ''})")

    _maybe_publish(results, opts)
    _write_manifest(work, src, opts, results)
    print(f"\nBitti. {len(results)} video: {work}")
    print("UYARI: Yayınlamadan ÖNCE videoyu elle izle ve telif/kaynak "
          "haklarını kabul ettiğini doğrula. Araç otomatik yayın YAPMAZ.")
    return results


def _whole_segment_clip(
    src: Path, start: float, seg_dur: float, idx: int, work: Path,
    opts: Options, *, single: bool, total_dur: float,
) -> ClipResult:
    """whole modunda tek bir parçayı hazırla (kes/kırp + altyazı)."""
    name = "video" if single else f"clip-{idx:02d}"
    base = work / name

    if single and not opts.vertical:
        media = src                      # tüm video, kırpma yok: kaynağı kullan
    elif opts.vertical:
        media = base.with_suffix(".mp4")
        clipper.cut_vertical(src, start, seg_dur, media)
    else:
        media = base.with_suffix(".mp4")
        clipper.cut_segment(src, start, seg_dur, media)

    final = media
    subtitled = False
    srt_out: str | None = None

    if opts.subtitles:
        srt_path = base.with_suffix(".srt")
        if subtitles.transcribe_to_srt(media, srt_path, opts.whisper_model, opts.lang):
            srt_out = str(srt_path)
            burned = base.with_name(f"{name}-sub.mp4")
            if subtitles.burn(
                media, srt_path, burned,
                font_size=opts.sub_size, margin_v=opts.sub_margin,
            ):
                final = burned
                subtitled = True

    # Hiç işlem olmadıysa (tek parça, altyazısız, dikey değil): kaynağı kopyala.
    if final == src:
        dst = base.with_suffix(".mp4")
        shutil.copy2(src, dst)
        final = dst

    title = (f"{opts.label} — {_mmss(total_dur)}" if single
             else f"{opts.label} #{idx} — {_mmss(start)}")
    return ClipResult(
        index=idx,
        file=str(final),
        start=round(start, 2),
        end=round(start + seg_dur, 2),
        peak=0.0,
        duration=round(seg_dur, 2),
        score=0.0,
        subtitled=subtitled,
        srt=srt_out,
        suggested_title=title,
    )


def _maybe_publish(results: list[ClipResult], opts: Options) -> None:
    """opts.publish ise her klibi YouTube'a (varsayılan private) yükle."""
    if not opts.publish or not results:
        return
    from . import publish as pub

    print(f"\n[Yayın] {len(results)} video YouTube'a yükleniyor "
          f"(gizlilik={opts.privacy}) ...")
    if opts.privacy == "public":
        print("  UYARI: public seçtin. Telif/spam riskini kabul ettiğini varsayıyorum.")
    for r in results:
        try:
            meta = pub.build_metadata(
                label=opts.label,
                srt_path=Path(r.srt) if r.srt else None,
            )
            url = pub.upload(
                Path(r.file), meta,
                client_secret=opts.client_secret,
                token_path=opts.token_path,
                privacy=opts.privacy,
            )
            r.youtube_url = url
            print(f"  #{r.index:02d} yüklendi -> {url}  (başlık: {meta['title']})")
        except Exception as e:
            print(f"  ! #{r.index:02d} yükleme başarısız: {str(e)[:300]}")
    print("  Not: videolar PRIVATE. YouTube Studio'da gözden geçirip elle yayınla.")


def _detect(src: Path, opts: Options) -> list[Moment]:
    if opts.mode == "match":
        if not opts.minutes:
            raise ValueError("match modu için --minutes gerekli (örn: 23,45+2,67).")
        return detect.detect_match(src, opts.minutes)
    return detect.detect_highlights(
        src, opts.count, scene_threshold=opts.scene_threshold
    )


def _make_clip(src: Path, m: Moment, idx: int, work: Path, opts: Options) -> ClipResult:
    base = work / f"clip-{idx:02d}"
    raw = base.with_suffix(".mp4")
    clipper.cut_vertical(src, m.start, m.duration, raw)

    final = raw
    subtitled = False
    srt_out: str | None = None
    if opts.subtitles:
        srt_path = base.with_suffix(".srt")
        if subtitles.transcribe_to_srt(raw, srt_path, opts.whisper_model, opts.lang):
            srt_out = str(srt_path)
            burned = base.with_name(f"clip-{idx:02d}-sub.mp4")
            if subtitles.burn(
                raw, srt_path, burned,
                font_size=opts.sub_size, margin_v=opts.sub_margin,
            ):
                final = burned
                subtitled = True

    title = f"{opts.label} #{idx} — {_mmss(m.peak)}"
    return ClipResult(
        index=idx,
        file=str(final),
        start=round(m.start, 2),
        end=round(m.end, 2),
        peak=round(m.peak, 2),
        duration=round(m.duration, 2),
        score=round(m.score, 5),
        subtitled=subtitled,
        srt=srt_out,
        suggested_title=title,
    )


def _mmss(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def _write_manifest(work: Path, src: Path, opts: Options, results: list[ClipResult]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "input": opts.source,
        "mode": opts.mode,
        "clip_count": len(results),
        "review_required": True,
        "clips": [asdict(r) for r in results],
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "MAÇ SHORTS — ÜRETİM RAPORU",
        f"Tarih    : {manifest['generated_at']}",
        f"Kaynak   : {opts.source}",
        f"Mod      : {opts.mode}",
        f"Klip     : {len(results)}",
        "",
        ">>> YAYINDAN ÖNCE ZORUNLU KONTROL <<<",
        "1. Her klibi izle: gol/an tam kadrajda mı, altyazı senkron mu?",
        "2. Telif riskini kabul ettiğini doğrula (maç görüntüsü = Content ID).",
        "3. Spam riski: aynı anda çok benzer klip atma sınırına dikkat.",
        "",
        "KLİPLER:",
    ]
    for r in results:
        lines.append(
            f"  #{r.index:02d}  {Path(r.file).name}  "
            f"[{_mmss(r.start)}-{_mmss(r.end)}]  {r.duration:.1f}sn  "
            f"{'altyazılı' if r.subtitled else 'altyazısız'}  "
            f"-> {r.suggested_title}"
        )
    (work / "review.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
