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
    headline = _first_sentence(caption) or src_title or _first_sentence(transcript)
    headline = re.sub(r"#\w+", "", headline).strip(" -|·–")
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
