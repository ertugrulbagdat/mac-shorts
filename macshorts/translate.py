"""SRT çevirisi: NVIDIA NIM (OpenAI uyumlu API) üzerinden İngilizce -> Türkçe.

Akıştaki yeri: faster-whisper İngilizce SRT'yi üretir (subtitles.transcribe_to_srt),
burada SADECE metin satırları çevrilir, ardından subtitles.burn çevrilmiş SRT'yi
videoya gömer.

Sözleşme — ZAMAN DAMGALARINA DOKUNULMAZ:
  * Cue sayısı, sırası, indeksleri ve "00:00:01,200 --> 00:00:03,400" satırları
    girdideki haliyle korunur; yalnızca metin değişir.
  * Bir cue çevrilemezse (model atlarsa / API patlarsa) o cue'nun İngilizce
    metni aynen kalır — altyazı hiç kaybolmaz.

Kurulum:
  pip install openai
  NVIDIA_API_KEY=nvapi-...   (build.nvidia.com -> API key)
  Anahtar proje kökündeki .env dosyasından da okunur (python-dotenv);
  ortamda zaten tanımlı bir değişken .env'i ezer.

Hattın geri kalanı gibi graceful degrade: openai kurulu değilse, anahtar yoksa
ya da API hata verirse uyarı basılır ve çeviri atlanır (İngilizce SRT gömülür).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

#: Proje kökündeki .env (macshorts/ paketinin bir üstü).
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

#: NVIDIA'nın barındırdığı model (OpenAI istemcisiyle çağrılır).
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
#: NVIDIA NIM'in OpenAI uyumlu uç noktası.
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
#: API anahtarı bu ortam değişkenlerinden ilk bulunandan okunur.
API_KEY_ENV = ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NGC_API_KEY")

#: Tek istekte gönderilecek cue sayısı. Bağlam (önceki/sonraki replik) çeviri
#: kalitesini artırır; çok büyük olursa model satır numaralarını karıştırır.
BATCH_SIZE = 20
MAX_RETRIES = 3
TIMEOUT = 90.0

_LANG_NAMES = {
    "en": "İngilizce", "tr": "Türkçe", "de": "Almanca", "fr": "Fransızca",
    "es": "İspanyolca", "it": "İtalyanca", "pt": "Portekizce", "ar": "Arapça",
    "ru": "Rusça", "nl": "Felemenkçe",
}

# Modelden satırları "[[3]] metin" biçiminde geri isteriz: hizalama bozulursa
# (model satır birleştirir/atlar) bunu TESPİT edip o cue'yu orijinal bırakırız.
_TAG_RE = re.compile(r"^\s*\[+\s*(\d+)\s*\]+[:.\-\s]*(.*)$")
_TS_RE = re.compile(r"-->")


class TranslateError(RuntimeError):
    pass


@dataclass
class Cue:
    """Tek bir SRT bloğu. `timing` satırı asla değiştirilmez."""

    index: str
    timing: str
    text: str


def lang_name(code: str) -> str:
    """Dil kodunu prompt'ta kullanılacak Türkçe adına çevir."""
    return _LANG_NAMES.get((code or "").lower().strip(), code or "bilinmeyen dil")


def parse_srt(content: str) -> list[Cue]:
    """SRT metnini cue listesine ayır. Zaman damgası satırı ham korunur."""
    cues: list[Cue] = []
    normalized = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n", normalized):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        idx = ""
        if not _TS_RE.search(lines[0]):
            idx = lines[0].strip()
            lines = lines[1:]
        if not lines or not _TS_RE.search(lines[0]):
            continue                      # zaman damgası yok: SRT bloğu değil
        timing = lines[0].strip()
        text = " ".join(ln.strip() for ln in lines[1:]).strip()
        cues.append(Cue(index=idx or str(len(cues) + 1), timing=timing, text=text))
    return cues


def render_srt(cues: list[Cue]) -> str:
    """Cue listesini SRT metnine geri yaz (indeks + zaman damgası + metin)."""
    out: list[str] = []
    for c in cues:
        out.append(c.index)
        out.append(c.timing)
        out.append(c.text)
        out.append("")
    return "\n".join(out)


def _load_dotenv() -> None:
    """Proje kökündeki .env'i ortama yükle (python-dotenv yoksa sessizce atla)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ENV_FILE, override=False)


def _api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    _load_dotenv()
    for name in API_KEY_ENV:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return None


def has_api_key(explicit: str | None = None) -> bool:
    """Çeviri için anahtar var mı (hat başında bir kez kontrol etmek için)."""
    return _api_key(explicit) is not None


def _client(api_key: str, base_url: str):
    from openai import OpenAI          # lazy: --no-translate ile gerekmez

    return OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT)


def _system_prompt(source_lang: str, target_lang: str) -> str:
    return (
        f"Sen profesyonel bir altyazı çevirmenisin. {source_lang} altyazı "
        f"satırlarını {target_lang} diline çevirirsin.\n"
        "KURALLAR:\n"
        "1. Girdideki her satır '[[numara]] metin' biçimindedir. Çıktıda AYNI "
        "numaraları, aynı sırada, aynı biçimde kullan.\n"
        "2. Satırları BİRLEŞTİRME, BÖLME, ATLAMA ve YENİ SATIR EKLEME. Girdide "
        "kaç numaralı satır varsa çıktıda tam olarak o kadar satır olacak.\n"
        f"3. Yalnızca metni çevir. Çeviri {target_lang} ve doğal konuşma dili "
        "olsun; altyazı olduğu için kısa ve akıcı tut.\n"
        "4. Özel isimleri (oyuncu, takım, stat, marka), sayıları ve skorları "
        "olduğu gibi bırak.\n"
        f"5. Bir satır zaten {target_lang} ise ya da çevrilecek bir şey yoksa "
        "(ünlem, ses efekti) olduğu gibi yaz.\n"
        "6. Açıklama, not, giriş cümlesi, tırnak ya da markdown EKLEME. "
        "Sadece numaralı satırları döndür."
    )


def _chat(client, model: str, system: str, user: str) -> str:
    """Sohbet tamamlama çağrısı; geçici hatalarda artan bekleme ile yeniden dene."""
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                top_p=0.9,
                max_tokens=2048,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:                      # ağ / oran sınırı / sunucu
            last = e
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 ** attempt)
    raise TranslateError(str(last))


def _parse_tagged(reply: str, numbers: list[int]) -> dict[int, str]:
    """Modelin '[[n]] metin' yanıtını numara -> metin eşlemesine çöz."""
    wanted = set(numbers)
    out: dict[int, str] = {}
    for raw in reply.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _TAG_RE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        text = m.group(2).strip().strip('"').strip()
        if n in wanted and text:
            out[n] = re.sub(r"\s+", " ", text)
    return out


def _translate_batch(
    client, model: str, system: str, batch: list[tuple[int, str]],
) -> dict[int, str]:
    """Bir grup cue'yu çevir. Dönen eşleme eksik olabilir (çağıran telafi eder)."""
    user = "\n".join(f"[[{n}]] {text}" for n, text in batch)
    reply = _chat(client, model, system, user)
    return _parse_tagged(reply, [n for n, _ in batch])


def translate_cues(
    cues: list[Cue],
    *,
    model: str = DEFAULT_MODEL,
    source_lang: str = "en",
    target_lang: str = "tr",
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
) -> tuple[list[Cue], int]:
    """Cue metinlerini çevir; zaman damgalarına dokunmaz.

    Dönen: (yeni cue listesi, çevrilen cue sayısı). Çevrilemeyen cue'lar
    orijinal metniyle kalır.
    """
    key = _api_key(api_key)
    if not key:
        raise TranslateError(
            "API anahtarı yok. " + " veya ".join(API_KEY_ENV) + " ortam "
            "değişkenini ayarla (build.nvidia.com -> API key)."
        )

    src_name, dst_name = lang_name(source_lang), lang_name(target_lang)
    system = _system_prompt(src_name, dst_name)
    client = _client(key, base_url)

    todo = [(i, c.text) for i, c in enumerate(cues) if c.text.strip()]
    translated: dict[int, str] = {}

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        try:
            got = _translate_batch(client, model, system, batch)
        except TranslateError as e:
            print(f"  ! Çeviri grubu başarısız ({start + 1}-{start + len(batch)}): "
                  f"{str(e)[:200]}")
            got = {}
        # Hizalama bozulduysa eksik cue'ları tek tek dene: bir satır numarası
        # karışması bütün grubu İngilizce bırakmasın.
        for n, text in [item for item in batch if item[0] not in got]:
            try:
                one = _translate_batch(client, model, system, [(n, text)])
            except TranslateError:
                one = {}
            if n in one:
                got[n] = one[n]
        translated.update(got)
        if verbose:
            done = min(start + batch_size, len(todo))
            print(f"      çeviri {done}/{len(todo)} cue")

    new_cues = [
        Cue(index=c.index, timing=c.timing, text=translated.get(i, c.text))
        for i, c in enumerate(cues)
    ]
    return new_cues, len(translated)


def translate_srt_file(
    src_srt: Path,
    dst_srt: Path,
    *,
    model: str = DEFAULT_MODEL,
    source_lang: str = "en",
    target_lang: str = "tr",
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
) -> bool:
    """SRT dosyasını çevirip `dst_srt`'e yaz. Başarılıysa True.

    Başarı ölçütü: en az bir cue gerçekten çevrildi. Aksi halde False döner ve
    çağıran orijinal (İngilizce) SRT ile devam eder — hat çökmez.
    """
    src_srt, dst_srt = Path(src_srt), Path(dst_srt)
    if not src_srt.exists():
        print(f"  ! Çevrilecek SRT bulunamadı: {src_srt}")
        return False

    cues = parse_srt(src_srt.read_text(encoding="utf-8", errors="replace"))
    if not cues:
        print("  ! SRT boş/çözümlenemedi, çeviri atlandı.")
        return False

    try:
        new_cues, n = translate_cues(
            cues, model=model, source_lang=source_lang, target_lang=target_lang,
            api_key=api_key, base_url=base_url, batch_size=batch_size,
            verbose=verbose,
        )
    except ImportError:
        print("  ! openai kurulu değil, çeviri atlandı (`pip install openai`).")
        return False
    except Exception as e:
        print(f"  ! Çeviri başarısız, İngilizce altyazı kullanılacak: {str(e)[:300]}")
        return False

    if n == 0:
        print("  ! Hiçbir cue çevrilemedi, İngilizce altyazı kullanılacak.")
        return False

    dst_srt.write_text(render_srt(new_cues), encoding="utf-8")
    if verbose:
        total = sum(1 for c in cues if c.text.strip())
        rest = "" if n == total else f" ({total - n} cue orijinal kaldı)"
        print(f"  ✓ Çeviri: {n}/{total} cue -> {lang_name(target_lang)}{rest} "
              f"[{dst_srt.name}]")
    return True


def main(argv: list[str] | None = None) -> int:
    """Tek başına kullanım: python -m macshorts.translate girdi.srt [çıktı.srt]"""
    import argparse

    p = argparse.ArgumentParser(
        prog="macshorts.translate",
        description="SRT altyazıyı NVIDIA NIM ile çevir (zaman damgaları korunur).",
    )
    p.add_argument("src", help="Girdi SRT")
    p.add_argument("dst", nargs="?", help="Çıktı SRT (varsayılan: <girdi>.tr.srt)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--from", dest="source_lang", default="en")
    p.add_argument("--to", dest="target_lang", default="tr")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    a = p.parse_args(argv)

    src = Path(a.src)
    dst = Path(a.dst) if a.dst else src.with_suffix(f".{a.target_lang}.srt")
    ok = translate_srt_file(
        src, dst, model=a.model, source_lang=a.source_lang,
        target_lang=a.target_lang, base_url=a.base_url, batch_size=a.batch_size,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
