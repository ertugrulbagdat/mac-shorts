"""Üretim hattı orkestrasyonu.

Akış (tasarım dokümanı, Recommended Approach):
  1. Girdi: URL/dosya + mod (highlights / match)
  2. yt-dlp ile indir (URL ise)
  3. Aday anları bul (özet: sahne+ses, tam maç: elle dakika + ses zirvesi)
  4. Her aday için 9:16 klip kes
  5. faster-whisper ile altyazı (varsayılan açık)
  5b. SRT'yi NVIDIA NIM ile Türkçe'ye çevir (varsayılan açık), çevrilmiş SRT gömülür
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
from . import translate as translate_mod  # Options.translate alanıyla çakışmasın
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
    srt: str | None                      # videoya gömülen SRT (çeviri açıksa Türkçe)
    suggested_title: str
    youtube_url: str | None = None
    srt_source: str | None = None        # whisper'ın ürettiği ham (çevrilmemiş) SRT
    translated: bool = False             # srt, srt_source'un çevirisi mi


@dataclass
class Options:
    source: str
    mode: str = "highlights"          # highlights | match | whole
    count: int = 5
    vertical: bool = False            # whole modunda 9:16'ya zorla (varsayılan: orijinal en-boy)
    smart_crop: bool = False          # 9:16 kırpmada aksiyonu takip et (varsayılan: merkez)
    horizontal: bool = False          # highlights/match: 9:16 kırpma yok, orijinal en-boy
    duration: float | None = None     # klip süresi (sn); None = mod varsayılanı (highlights 20)
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
    translate: bool = True            # SRT'yi NVIDIA NIM ile çevir (anahtar yoksa atlanır)
    translate_from: str = "en"        # whisper SRT'sinin dili
    translate_to: str = "tr"          # gömülecek altyazının dili
    translate_model: str = translate_mod.DEFAULT_MODEL
    translate_base_url: str = translate_mod.DEFAULT_BASE_URL
    publish: bool = False             # YouTube'a yarı-otomatik yükleme
    privacy: str = "private"          # private | unlisted | public (varsayılan private)
    client_secret: Path = Path("client_secret.json")
    token_path: Path = Path("youtube_token.json")


def _sub_stage_label(opts: Options) -> str:
    """İlerleme satırı için altyazı/çeviri aşama etiketi."""
    if not opts.subtitles:
        return ""
    if opts.translate:
        return f" + altyazı ({translate_mod.lang_name(opts.translate_to)} çeviri)"
    return " + altyazı"


def _check_translate_ready(opts: Options) -> None:
    """Çeviri ön koşullarını hat başında BİR KEZ doğrula; eksikse çeviriyi kapat.

    Böylece her klip için aynı uyarıyı basmayız ve manifest'te çevirinin
    gerçekten çalışmadığı dürüstçe görünür.
    """
    if not (opts.subtitles and opts.translate):
        return
    try:
        import openai  # noqa: F401
    except ImportError:
        print("  ! openai kurulu değil -> altyazı çevirisi kapatıldı "
              "(`pip install openai`).")
        opts.translate = False
        return
    if not translate_mod.has_api_key():
        print(f"  ! {translate_mod.API_KEY_ENV[0]} ayarlı değil -> altyazı "
              "çevirisi kapatıldı; whisper altyazısı olduğu gibi gömülecek.")
        opts.translate = False


def run(opts: Options) -> list[ClipResult]:
    out_dir = Path(opts.out_dir)
    work = out_dir / datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    work.mkdir(parents=True, exist_ok=True)

    _check_translate_ready(opts)

    print(f"[1/4] Kaynak alınıyor: {opts.source}")
    src, source_meta = download.fetch(opts.source, work / "_source")
    print(f"      -> {src}")

    if opts.mode == "whole":
        return _process_whole(src, work, opts, source_meta)

    print(f"[2/4] Anlar tespit ediliyor (mod={opts.mode}) ...")
    moments = _detect(src, opts)
    if not moments:
        print("      ! Hiç aday an bulunamadı.")
        return []
    print(f"      -> {len(moments)} aday an")

    fmt = "yatay" if opts.horizontal else "9:16"
    print(f"[3/4] Klipler kesiliyor ({fmt}){_sub_stage_label(opts)} ...")
    results: list[ClipResult] = []
    multi = len(moments) > 1
    for i, m in enumerate(moments, start=1):
        res = _make_clip(src, m, i, work, opts, source_meta, part=i if multi else None)
        results.append(res)
        print(f"      [{i}/{len(moments)}] {res.file} "
              f"({res.duration:.1f}sn{', altyazılı' if res.subtitled else ''})")

    _maybe_publish(results, opts, source_meta)

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


def _process_whole(
    src: Path, work: Path, opts: Options, source_meta: dict | None = None,
) -> list[ClipResult]:
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
    print(f"[3/3] Parça(lar) işleniyor{_sub_stage_label(opts)} ...")

    results: list[ClipResult] = []
    for i, (start, seg_dur) in enumerate(segments, start=1):
        res = _whole_segment_clip(
            src, start, seg_dur, i, work, opts, source_meta,
            single=not multi, total_dur=dur,
        )
        results.append(res)
        print(f"      [{i}/{len(segments)}] {Path(res.file).name} "
              f"({res.duration:.1f}sn{', altyazılı' if res.subtitled else ''})")

    _maybe_publish(results, opts, source_meta)
    _write_manifest(work, src, opts, results)
    print(f"\nBitti. {len(results)} video: {work}")
    print("UYARI: Yayınlamadan ÖNCE videoyu elle izle ve telif/kaynak "
          "haklarını kabul ettiğini doğrula. Araç otomatik yayın YAPMAZ.")
    return results


def _whole_segment_clip(
    src: Path, start: float, seg_dur: float, idx: int, work: Path,
    opts: Options, source_meta: dict | None = None, *, single: bool, total_dur: float,
) -> ClipResult:
    """whole modunda tek bir parçayı hazırla (kes/kırp + altyazı)."""
    name = "video" if single else f"clip-{idx:02d}"
    base = work / name

    if single and not opts.vertical:
        media = src                      # tüm video, kırpma yok: kaynağı kullan
    elif opts.vertical:
        media = base.with_suffix(".mp4")
        clipper.cut_vertical(src, start, seg_dur, media, smart=opts.smart_crop)
    else:
        media = base.with_suffix(".mp4")
        clipper.cut_segment(src, start, seg_dur, media)

    sub = _apply_subtitles(media, base, opts) if opts.subtitles else _SubResult(
        final=media, subtitled=False,
    )
    final = sub.final

    # Hiç işlem olmadıysa (tek parça, altyazısız, dikey değil): kaynağı kopyala.
    if final == src:
        dst = base.with_suffix(".mp4")
        shutil.copy2(src, dst)
        final = dst

    fallback = (f"{opts.label} — {_mmss(total_dur)}" if single
                else f"{opts.label} #{idx} — {_mmss(start)}")
    title = _suggested_title(
        opts, sub.srt, source_meta, part=None if single else idx, fallback=fallback,
    )
    return ClipResult(
        index=idx,
        file=str(final),
        start=round(start, 2),
        end=round(start + seg_dur, 2),
        peak=0.0,
        duration=round(seg_dur, 2),
        score=0.0,
        subtitled=sub.subtitled,
        srt=sub.srt,
        suggested_title=title,
        srt_source=sub.srt_source,
        translated=sub.translated,
    )


@dataclass
class _SubResult:
    """_apply_subtitles çıktısı: gömülecek video + altyazı dosyaları."""

    final: Path
    subtitled: bool
    srt: str | None = None               # gömülen SRT (çeviri başarılıysa Türkçe)
    srt_source: str | None = None        # whisper'ın ham SRT'si
    translated: bool = False


def _apply_subtitles(media: Path, base: Path, opts: Options) -> _SubResult:
    """Altyazı zinciri: whisper SRT -> (NVIDIA) çeviri -> ffmpeg ile gömme.

    Her adım bağımsız olarak düşebilir: transkript olmazsa altyazı yok, çeviri
    olmazsa whisper'ın İngilizce SRT'si gömülür, gömme olmazsa SRT yan dosya
    olarak kalır. Hat hiçbir durumda çökmez.
    """
    srt_path = base.with_suffix(".srt")
    if not subtitles.transcribe_to_srt(media, srt_path, opts.whisper_model, opts.lang):
        return _SubResult(final=media, subtitled=False)

    burn_srt = srt_path
    srt_out = str(srt_path)
    translated = False

    if opts.translate:
        tr_path = base.with_name(f"{base.name}.{opts.translate_to}.srt")
        if translate_mod.translate_srt_file(
            srt_path, tr_path,
            model=opts.translate_model,
            source_lang=opts.translate_from,
            target_lang=opts.translate_to,
            base_url=opts.translate_base_url,
        ):
            burn_srt, srt_out, translated = tr_path, str(tr_path), True

    burned = base.with_name(f"{base.name}-sub.mp4")
    if subtitles.burn(
        media, burn_srt, burned,
        font_size=opts.sub_size, margin_v=opts.sub_margin,
    ):
        return _SubResult(
            final=burned, subtitled=True, srt=srt_out,
            srt_source=str(srt_path), translated=translated,
        )
    return _SubResult(
        final=media, subtitled=False, srt=srt_out,
        srt_source=str(srt_path), translated=translated,
    )


def _maybe_publish(
    results: list[ClipResult], opts: Options, source_meta: dict | None = None,
) -> None:
    """opts.publish ise her klibi YouTube'a (varsayılan private) yükle."""
    if not opts.publish or not results:
        return
    from . import publish as pub

    multi = len(results) > 1
    print(f"\n[Yayın] {len(results)} video YouTube'a yükleniyor "
          f"(gizlilik={opts.privacy}) ...")
    if opts.privacy == "public":
        print("  UYARI: public seçtin. Telif/spam riskini kabul ettiğini varsayıyorum.")
    for r in results:
        try:
            meta = pub.build_metadata(
                label=opts.label,
                srt_path=Path(r.srt) if r.srt else None,
                source=source_meta,
                part=(r.index if multi else None),
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
        return detect.detect_match(src, opts.minutes, clip_len=opts.duration)
    return detect.detect_highlights(
        src, opts.count, scene_threshold=opts.scene_threshold, clip_len=opts.duration,
    )


def _make_clip(
    src: Path, m: Moment, idx: int, work: Path, opts: Options,
    source_meta: dict | None = None, *, part: int | None = None,
) -> ClipResult:
    base = work / f"clip-{idx:02d}"
    raw = base.with_suffix(".mp4")
    if opts.horizontal:
        clipper.cut_segment(src, m.start, m.duration, raw)
    else:
        clipper.cut_vertical(src, m.start, m.duration, raw, smart=opts.smart_crop)

    sub = _apply_subtitles(raw, base, opts) if opts.subtitles else _SubResult(
        final=raw, subtitled=False,
    )

    title = _suggested_title(
        opts, sub.srt, source_meta, part=part,
        fallback=f"{opts.label} #{idx} — {_mmss(m.peak)}",
    )
    return ClipResult(
        index=idx,
        file=str(sub.final),
        start=round(m.start, 2),
        end=round(m.end, 2),
        peak=round(m.peak, 2),
        duration=round(m.duration, 2),
        score=round(m.score, 5),
        subtitled=sub.subtitled,
        srt=sub.srt,
        suggested_title=title,
        srt_source=sub.srt_source,
        translated=sub.translated,
    )


def _suggested_title(
    opts: Options, srt_out: str | None, source_meta: dict | None,
    *, part: int | None, fallback: str,
) -> str:
    """Manifest için önerilen başlık — YAYINDA kullanılacakla AYNI mantık.

    publish.build_metadata caption + klip transkriptinden anlamlı başlık üretir
    (gol/ofsayt/... farkındalıklı). Böylece review.txt'te gördüğün başlık,
    --publish ile yüklenecek başlıkla birebir aynı olur. Herhangi bir sorunda
    eski 'label #idx — mm:ss' biçimine düşeriz.
    """
    try:
        from . import publish as pub

        meta = pub.build_metadata(
            label=opts.label,
            srt_path=Path(srt_out) if srt_out else None,
            source=source_meta,
            part=part,
        )
        title = (meta.get("title") or "").strip()
        return title or fallback
    except Exception:
        return fallback


def _mmss(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def _write_manifest(work: Path, src: Path, opts: Options, results: list[ClipResult]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "input": opts.source,
        "mode": opts.mode,
        "format": "horizontal" if opts.horizontal else "vertical",
        "clip_count": len(results),
        "review_required": True,
        "subtitle_language": (
            opts.translate_to if any(r.translated for r in results)
            else (opts.lang or "auto")
        ),
        "translation": {
            "enabled": opts.translate,
            "provider": "nvidia-nim",
            "model": opts.translate_model,
            "from": opts.translate_from,
            "to": opts.translate_to,
            "applied": sum(1 for r in results if r.translated),
        },
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
        "2. Altyazı çevirisi makine çevirisidir (NVIDIA/Llama): isim, skor ve "
        "olay doğru mu, kontrol et.",
        "3. Telif riskini kabul ettiğini doğrula (maç görüntüsü = Content ID).",
        "4. Spam riski: aynı anda çok benzer klip atma sınırına dikkat.",
        "",
        "KLİPLER:",
    ]
    for r in results:
        if not r.subtitled:
            sub_state = "altyazısız"
        elif r.translated:
            sub_state = f"altyazılı ({opts.translate_to.upper()} çeviri)"
        else:
            sub_state = "altyazılı (çevrilmedi)" if opts.translate else "altyazılı"
        lines.append(
            f"  #{r.index:02d}  {Path(r.file).name}  "
            f"[{_mmss(r.start)}-{_mmss(r.end)}]  {r.duration:.1f}sn  "
            f"{sub_state}  -> {r.suggested_title}"
        )
    (work / "review.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
