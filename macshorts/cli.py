"""Komut satırı arayüzü."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import Options, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="macshorts",
        description="YouTube maç videosundan 9:16 Shorts klipleri üretir. "
                    "Otomatik YAYIN YAPMAZ; klipleri elle onaylarsın.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube linki")
    src.add_argument("--file", help="Yerel video dosyası (indirmeyi atlar)")

    p.add_argument(
        "--mode", choices=["highlights", "match"], default="highlights",
        help="highlights: özet videoda otomatik sahne+ses tespiti. "
             "match: tam maçta elle gol dakikaları (--minutes).",
    )
    p.add_argument(
        "--count", type=int, default=5,
        help="highlights modunda üretilecek klip sayısı (varsayılan 5).",
    )
    p.add_argument(
        "--minutes",
        help="match modu için gol dakikaları, örn: 23,45+2,67",
    )
    p.add_argument("--out", default="output", help="Çıktı klasörü (varsayılan output/).")
    p.add_argument(
        "--no-subtitles", action="store_true",
        help="Altyazı üretmeyi atla (faster-whisper kullanılmaz).",
    )
    p.add_argument(
        "--whisper-model", default="small",
        help="faster-whisper model adı (tiny/base/small/medium). Varsayılan small.",
    )
    p.add_argument(
        "--lang",
        help="Altyazı dili (örn: tr, en). Boş bırakılırsa otomatik algılanır.",
    )
    p.add_argument(
        "--scene-threshold", type=float, default=0.35,
        help="highlights sahne tespiti eşiği (0-1, varsayılan 0.35).",
    )
    p.add_argument(
        "--sub-size", type=int, default=12,
        help="Altyazı font boyutu (küçük sayı; varsayılan 12).",
    )
    p.add_argument(
        "--sub-margin", type=int, default=45,
        help="Altyazı alt boşluğu; küçüldükçe altyazı daha aşağı iner "
             "(varsayılan 45, eski 120 ekran ortasıydı).",
    )
    p.add_argument(
        "--label", default="klip",
        help="Önerilen başlık öneki (örn: 'Dünya Kupası gol').",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "match" and not args.minutes:
        print("Hata: match modu için --minutes gerekli (örn: 23,45+2,67).",
              file=sys.stderr)
        return 2

    opts = Options(
        source=args.url or args.file,
        mode=args.mode,
        count=args.count,
        minutes=args.minutes,
        out_dir=Path(args.out),
        subtitles=not args.no_subtitles,
        whisper_model=args.whisper_model,
        lang=args.lang,
        scene_threshold=args.scene_threshold,
        label=args.label,
        sub_size=args.sub_size,
        sub_margin=args.sub_margin,
    )

    try:
        results = run(opts)
    except KeyboardInterrupt:
        print("\nİptal edildi.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
        return 1

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
