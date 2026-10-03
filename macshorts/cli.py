"""Komut satırı arayüzü."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import translate
from .pipeline import Options, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="macshorts",
        description="YouTube maç videosundan 9:16 Shorts klipleri üretir. "
                    "Otomatik YAYIN YAPMAZ; klipleri elle onaylarsın.",
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--url", help="YouTube/Instagram linki")
    src.add_argument("--file", help="Yerel video dosyası (indirmeyi atlar)")

    p.add_argument(
        "--mode", choices=["highlights", "match", "whole"], default="highlights",
        help="highlights: özet videoda otomatik sahne+ses tespiti. "
             "match: tam maçta elle gol dakikaları (--minutes). "
             "whole: videoyu parçalamadan indir + altyazı ekle (reel için).",
    )
    p.add_argument(
        "--vertical", action="store_true",
        help="whole modunda tüm videoyu 9:16'ya kırp (varsayılan: orijinal en-boy).",
    )
    p.add_argument(
        "--horizontal", action="store_true",
        help="highlights/match modunda klipleri 9:16'ya kırpma; orijinal yatay "
             "(16:9) en-boyu koru.",
    )
    p.add_argument(
        "--duration", type=float, default=None,
        help="highlights/match klip süresi (sn), örn: --duration 60. "
             "Varsayılan highlights'ta 20, match'te 16.",
    )
    p.add_argument(
        "--smart-crop", action="store_true",
        help="9:16 kırpmada sabit merkez yerine aksiyonu (top/oyun) takip eden "
             "kayan pencere kullan. Hareket sinyali yoksa merkeze düşer.",
    )
    p.add_argument(
        "--short-len", type=float, default=60.0,
        help="whole modunda bu süreden (sn) uzun video parçalara bölünür "
             "(varsayılan 60). Kısa videolar tek parça kalır.",
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
    tr = p.add_argument_group(
        "Altyazı çevirisi (NVIDIA NIM — varsayılan AÇIK)",
        "Whisper SRT'sini NVIDIA'nın barındırdığı Llama modeliyle Türkçe'ye "
        "çevirir, zaman damgalarına dokunmaz ve çevrilmiş SRT videoya gömülür. "
        "NVIDIA_API_KEY ortam değişkeni gerekir; yoksa uyarı verilip İngilizce "
        "altyazı gömülür.",
    )
    tr.add_argument(
        "--no-translate", action="store_true",
        help="Çeviriyi atla, whisper'ın ürettiği altyazıyı doğrudan göm.",
    )
    tr.add_argument(
        "--translate-to", default="tr",
        help="Hedef altyazı dili kodu (varsayılan tr).",
    )
    tr.add_argument(
        "--translate-from", default="en",
        help="Whisper SRT'sinin dili (varsayılan en).",
    )
    tr.add_argument(
        "--translate-model", default=translate.DEFAULT_MODEL,
        help=f"NVIDIA NIM model adı (varsayılan {translate.DEFAULT_MODEL}).",
    )
    tr.add_argument(
        "--translate-base-url", default=translate.DEFAULT_BASE_URL,
        help="OpenAI uyumlu uç nokta (varsayılan NVIDIA NIM).",
    )
    pub = p.add_argument_group("YouTube yayını (yarı-otomatik)")
    pub.add_argument(
        "--login", action="store_true",
        help="Tek seferlik YouTube OAuth girişi yap (tarayıcı açılır), token "
             "kaydet ve çık. İNTERAKTİF terminalde çalıştır. URL gerekmez.",
    )
    pub.add_argument(
        "--publish", action="store_true",
        help="Üretilen videoları YouTube'a yükle (varsayılan PRIVATE). "
             "Başlık/açıklama otomatik üretilir; sen Studio'da yayınlarsın.",
    )
    pub.add_argument(
        "--privacy", choices=["private", "unlisted", "public"], default="private",
        help="Yükleme gizliliği (varsayılan private).",
    )
    pub.add_argument(
        "--client-secret", default="client_secret.json",
        help="Google OAuth client_secret.json yolu.",
    )
    pub.add_argument(
        "--token", default="youtube_token.json",
        help="OAuth token'ın saklanacağı/okunacağı yol.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Tek seferlik OAuth girişi: URL/dosya gerekmez.
    if args.login:
        from . import publish
        try:
            publish.login(Path(args.client_secret), Path(args.token))
            return 0
        except Exception as e:
            print(f"Giriş hatası: {e}", file=sys.stderr)
            return 1

    if not (args.url or args.file):
        print("Hata: --url veya --file gerekli (ya da tek seferlik giriş için --login).",
              file=sys.stderr)
        return 2

    if args.duration is not None and args.duration <= 0:
        print("Hata: --duration pozitif bir saniye değeri olmalı.", file=sys.stderr)
        return 2

    if args.horizontal and args.vertical:
        print("Hata: --horizontal ve --vertical birlikte kullanılamaz.", file=sys.stderr)
        return 2

    if args.mode == "match" and not args.minutes:
        print("Hata: match modu için --minutes gerekli (örn: 23,45+2,67).",
              file=sys.stderr)
        return 2

    opts = Options(
        source=args.url or args.file,
        mode=args.mode,
        vertical=args.vertical,
        smart_crop=args.smart_crop,
        horizontal=args.horizontal,
        duration=args.duration,
        short_len=args.short_len,
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
        translate=not args.no_translate,
        translate_from=args.translate_from,
        translate_to=args.translate_to,
        translate_model=args.translate_model,
        translate_base_url=args.translate_base_url,
        publish=args.publish,
        privacy=args.privacy,
        client_secret=Path(args.client_secret),
        token_path=Path(args.token),
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
