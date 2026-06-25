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


def build_metadata(
    *,
    label: str,
    srt_path: Path | None,
    extra_tags: list[str] | None = None,
) -> dict:
    """Transkriptten başlık/açıklama/etiket üret.

    Title <= 100 karakter (YouTube sınırı). Açıklama transkript özeti + hashtag.
    """
    transcript = _read_transcript(srt_path)
    first = ""
    if transcript:
        # İlk cümle (nokta/ünlem/soru) ya da ilk ~70 karakter.
        m = re.split(r"(?<=[.!?])\s", transcript, maxsplit=1)
        first = m[0] if m else transcript
        first = first[:70].strip()

    base = label.strip() or "Short"
    title = f"{base} | {first}" if first else base
    title = title[:90].rstrip(" |")
    if "#shorts" not in title.lower() and len(title) <= 82:
        title = f"{title} #Shorts"

    tags = ["shorts", "short"]
    tags += [t.lower() for t in base.split() if len(t) > 2]
    if extra_tags:
        tags += [t.lower() for t in extra_tags]
    # benzersiz + sınırlı
    seen: set[str] = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))][:15]

    hashtags = " ".join(f"#{re.sub(r'[^0-9a-zçğıöşü]', '', t.lower())}" for t in tags[:5] if t)
    desc_parts = []
    if transcript:
        desc_parts.append(transcript[:300].strip())
    desc_parts.append(hashtags)
    description = "\n\n".join(p for p in desc_parts if p)

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
