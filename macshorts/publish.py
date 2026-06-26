"""YouTube'a YARI-OTOMATİK yükleme: video private/taslak yüklenir, başlık/
açıklama/etiket transkriptten otomatik üretilir. Son "yayınla" kararı insanda.

Tasarım gereği güvenlik kapısı: privacy varsayılanı 'private'. Otomatik public
yayın yok (telif/spam riski). Kullanıcı YouTube Studio'da gözden geçirip elle
yayınlar.

Bir kerelik kurulum (README'de detaylı):
  1. Google Cloud projesi + YouTube Data API v3 etkin.
  2. OAuth "Desktop app" kimlik bilgisi -> client_secret.json indir.
  3. İlk --publish çalıştırmasında tarayıcıda izin ver; token saklanır.

Google kütüphaneleri yalnızca burada (lazy) import edilir; --publish
kullanılmadıkça projenin geri kalanı bu bağımlılıklar olmadan çalışır.
"""
from __future__ import annotations

import re
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
# YouTube kategori 17 = Sports. Spor dışı için 24 (Entertainment) makul varsayılan.
DEFAULT_CATEGORY = "17"


class PublishError(RuntimeError):
    pass


def _read_transcript(srt_path: Path | None) -> str:
    """SRT'den düz metni çıkar (indeks ve zaman damgası satırlarını atla)."""
    if not srt_path or not Path(srt_path).exists():
        return ""
    lines: list[str] = []
    for raw in Path(srt_path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.isdigit() or "-->" in s:
            continue
        lines.append(s)
    return " ".join(lines).strip()


def _oneline(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean(text: str | None) -> str:
    """HTML kaçışlarını çöz, satır sonlarını sadeleştir, kenar boşluklarını al."""
    import html

    if not text:
        return ""
    text = html.unescape(text).replace("\r", "\n")
    # 3+ ardışık satır sonunu 2'ye indir
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _first_sentence(text: str | None) -> str:
    t = _oneline(text)
    if not t:
        return ""
    m = re.split(r"(?<=[.!?])\s", t, maxsplit=1)
    return (m[0] if m else t).strip()


# Futbol olay kelimeleri (TR + EN). Bir cümlede geçiyorsa o cümle başlık için
# çok daha değerli: izleyici "ne oldu"yu başlıkta görür. Kelime SINIRIYLA
# eşleşir (aşağıda) — "gol" "golf"e, "post" "poster"a takılmasın diye. Türkçe'de
# aşırı yaygın "var" (VAR hakemi) bilinçli DIŞARIDA: her cümlede eşleşirdi.
_EVENT_KEYWORDS = (
    "gol", "golü", "goal", "penaltı", "penalty", "ofsayt", "offside",
    "kırmızı kart", "red card", "sent off", "kurtarış", "kurtardı", "kaleci",
    "save", "saved", "saves", "keeper", "asist", "assist", "frikik", "free kick",
    "korner", "corner", "kaçırdı", "missed", "crossbar", "direk",
    "muhteşem", "harika", "incredible", "stunning", "brilliant",
)
_EVENT_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _EVENT_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _has_event(text: str) -> bool:
    return bool(_EVENT_RE.search(text or ""))


def _is_garbled(text: str) -> bool:
    """ASR fragmanı başlık olamayacak kadar YAPISAL olarak bozuk mu?

    Hızlı spiker anlatımı faster-whisper'da bazen anlamsız çıkar. Heuristik:
    harf oranı düşükse, çok kısa "kelimeler" baskınsa ya da aynı harf üst üste
    tekrarlıyorsa başlığa koymayız. Not: bu yapısal kontrol UZUNLUĞA bakmaz;
    "Gol!" gibi kısa ama temiz olay cümlelerini eler düşürmemek için.
    """
    t = (text or "").strip()
    if len(t) < 4:
        return True
    letters = sum(c.isalpha() for c in t)
    if letters / max(len(t), 1) < 0.55:
        return True
    words = re.findall(r"\S+", t)
    if not words:
        return True
    # Tek-iki harflik token'lar baskınsa (parçalı/bozuk ASR imzası)
    tiny = sum(1 for w in words if len(re.sub(r"[^0-9a-zçğıöşü]", "", w.lower())) <= 2)
    if tiny / len(words) > 0.5:
        return True
    # Aynı harfin 4+ kez ardışık tekrarı (ör. "aaaa", "lllll")
    if re.search(r"(.)\1{3,}", t):
        return True
    return False


def _score_sentence(s: str) -> float:
    """Başlık adayı cümleyi puanla. Yüksek = daha iyi başlık."""
    s = s.strip()
    n = len(s)
    if n == 0 or _is_garbled(s):
        return -1.0
    score = 0.0
    has_event = _has_event(s)
    if has_event:
        score += 3.0          # "ne oldu" başlıkta: en değerli sinyal
    if s.endswith("!"):
        score += 1.0          # heyecanlı kapanış
    # Uzunluk tatlı noktası: 18-75 karakter okunur bir başlık.
    if 18 <= n <= 75:
        score += 1.0
    elif n < 12 and not has_event:
        score -= 1.5          # kısa + olaysız = zayıf (ama "Gol!" cezalanmaz)
    return score


def _best_headline(text: str | None) -> str:
    """Transkriptten en başlık-değeri yüksek cümleyi seç (kalite kapısından geçen).

    Hiçbir cümle yeterince iyi/temiz değilse boş döner — çağıran caption/kaynak
    başlığına düşer. Bu, eski "parça transkriptine hiç dokunma" davranışının
    yerini alır: temizse kullan, bozuksa atla.
    """
    t = _oneline(text)
    if not t:
        return ""
    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    if not sentences:
        sentences = [t]
    best, best_score = "", 0.0
    for s in sentences:
        sc = _score_sentence(s)
        if sc > best_score:
            best, best_score = s, sc
    # Olay kelimesi yoksa (skor düşük) ama tek temiz cümle varsa onu kabul et.
    if not best and not _is_garbled(sentences[0]):
        best = _first_sentence(t)
    return best.strip(" -|·–")


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _hashtags(text: str | None) -> list[str]:
    return [h.lower() for h in re.findall(r"#(\w+)", text or "")]


def build_metadata(
    *,
    label: str,
    srt_path: Path | None,
    source: dict | None = None,
    extra_tags: list[str] | None = None,
    part: int | None = None,
) -> dict:
    """Kaynak caption + transkript + etiketten anlamlı başlık/açıklama üret.

    Öncelik sırası (başlık için): kaynak caption ilk cümlesi > kaynak başlığı >
    transkript ilk cümlesi. Açıklama: caption (tam) + transkript özeti (varsa) +
    kaynak künyesi + hashtag. Title <= 100 (YouTube sınırı).
    """
    source = source or {}
    caption = _clean(source.get("caption"))
    src_title = _oneline(source.get("title"))
    uploader = (source.get("uploader") or "").strip()
    transcript = _read_transcript(srt_path)

    # En anlamlı kısa başlık metni (hashtagsiz).
    # Önce klibin KENDİ transkriptinden en başlık-değeri yüksek (gol/ofsayt/...
    # içeren, kalite kapısından geçen) cümleyi dene — her klibe ayırt edici bir
    # başlık verir. Hızlı spiker anlatımı bozuksa _best_headline boş döner ve
    # caption / kaynak başlığına düşeriz (eski "transkripte hiç dokunma"
    # davranışı yerine: temizse kullan, bozuksa atla).
    clip_headline = _best_headline(transcript)
    if part is not None:
        # Çok parçalı klip: parça transkripti > caption ilk cümlesi > kaynak başlığı.
        headline = clip_headline or _first_sentence(caption) or src_title
    else:
        headline = clip_headline or _first_sentence(caption) or src_title or _first_sentence(transcript)
    headline = re.sub(r"#\w+", "", headline or "").strip(" -|·–")
    headline = _oneline(headline)

    base = label.strip()
    pieces = [p for p in (base, headline) if p and p.lower() != base.lower()]
    if base and base not in pieces:
        pieces.insert(0, base)
    title = " | ".join(pieces) if pieces else (headline or "Short")
    if part:
        title = f"{title} (Bölüm {part})"
    title = _truncate(title, 90)
    if "#shorts" not in title.lower() and len(title) <= 82:
        title = f"{title} #Shorts"

    # Etiketler: caption hashtag'leri + etiket kelimeleri + kaynak etiketleri.
    tag_pool = ["shorts", "short"]
    tag_pool += _hashtags(caption)
    tag_pool += [w.lower() for w in re.findall(r"\w+", base) if len(w) > 2]
    tag_pool += [str(t).lower().replace(" ", "") for t in (source.get("tags") or [])]
    if extra_tags:
        tag_pool += [t.lower() for t in extra_tags]
    seen: set[str] = set()
    tags = [t for t in tag_pool if t and not (t in seen or seen.add(t))][:15]

    # Açıklama blokları.
    blocks: list[str] = []
    if caption:
        blocks.append(_truncate(caption, 500))
    elif transcript:
        blocks.append(_truncate(transcript, 350))
    # Konuşma transkripti caption'da yoksa ekle (ek bağlam).
    if caption and transcript and _oneline(transcript)[:50].lower() not in caption.lower():
        blocks.append("📝 " + _truncate(transcript, 220))
    if uploader:
        blocks.append(f"Kaynak: @{uploader}")
    hashtag_line = " ".join(
        f"#{re.sub(r'[^0-9a-zçğıöşü]', '', t)}" for t in tags[:6] if t
    )
    if hashtag_line:
        blocks.append(hashtag_line)
    description = "\n\n".join(b for b in blocks if b).strip()

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "categoryId": DEFAULT_CATEGORY,
    }


def _get_credentials(client_secret: Path, token_path: Path):
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise PublishError(
            "Yayın için Google kütüphaneleri gerekli:\n"
            "  pip install google-api-python-client google-auth-oauthlib"
        ) from e

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not client_secret.exists():
            raise PublishError(
                f"client_secret.json bulunamadı: {client_secret}\n"
                "Google Cloud'da OAuth 'Desktop app' kimliği oluşturup indir "
                "(README → Yayın kurulumu)."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
        creds = flow.run_local_server(port=0)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def login(client_secret: Path, token_path: Path) -> None:
    """Tek seferlik OAuth: tarayıcıda izin al, token'ı kaydet.

    İNTERAKTİF terminalde çalıştırılmalı (tarayıcı açılır). Sonraki --publish
    çalıştırmaları token'ı kullanır, tekrar giriş gerekmez.
    """
    _get_credentials(client_secret, token_path)
    print(f"Giriş başarılı. Token kaydedildi: {token_path}")
    print("Artık --publish ile yükleme yapabilirsin (tekrar giriş gerekmez).")


def upload(
    video_path: Path,
    metadata: dict,
    *,
    client_secret: Path,
    token_path: Path,
    privacy: str = "private",
) -> str:
    """Videoyu YouTube'a yükle. Yüklenen videonun URL'ini döndür."""
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as e:
        raise PublishError(
            "Yayın için Google kütüphaneleri gerekli:\n"
            "  pip install google-api-python-client google-auth-oauthlib"
        ) from e

    creds = _get_credentials(client_secret, token_path)
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": metadata["title"],
            "description": metadata["description"],
            "tags": metadata.get("tags", []),
            "categoryId": metadata.get("categoryId", DEFAULT_CATEGORY),
        },
        "status": {
            "privacyStatus": privacy,          # varsayılan: private
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        _status, response = request.next_chunk()
    vid = response["id"]
    return f"https://youtu.be/{vid}"
