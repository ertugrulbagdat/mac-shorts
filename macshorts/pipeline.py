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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import clip as clipper
from . import detect, download, subtitles
from .detect import Moment


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


@dataclass
class Options:
    source: str
    mode: str = "highlights"          # highlights | match
    count: int = 5
    minutes: str | None = None        # match modu için "23,45+2"
    out_dir: Path = Path("output")
    subtitles: bool = True
    whisper_model: str = "small"
    lang: str | None = None
    scene_threshold: float = 0.35
    label: str = "klip"               # önerilen başlık öneki
    sub_size: int = 12                # altyazı font boyutu (libass SRT tuvali); küçük sayı
    sub_margin: int = 45              # altyazı alt boşluğu; küçüldükçe daha aşağı


def run(opts: Options) -> list[ClipResult]:
    out_dir = Path(opts.out_dir)
    work = out_dir / datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    work.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Kaynak alınıyor: {opts.source}")
    src = download.fetch(opts.source, work / "_source")
    print(f"      -> {src}")

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

    print("[4/4] Manifest yazılıyor ...")
    _write_manifest(work, src, opts, results)
    print(f"\nBitti. {len(results)} klip: {work}")
    print("UYARI: Yayınlamadan ÖNCE klipleri elle izle ve telif riskini kabul "
          "ettiğini doğrula. Araç otomatik yayın YAPMAZ.")
    return results


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
