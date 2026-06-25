# macshorts

YouTube maç videosundan heyecanlı anları bulup **9:16 Shorts** klipleri üreten
komut satırı aracı. Tasarım dokümanı: `office-hours` seansından üretilen
`ertug-master-design-*.md`.

> **Bu araç otomatik YAYIN YAPMAZ.** Sadece klip üretir. Paylaşma kararı ve
> telif sorumluluğu sana aittir. Maç görüntüleri Content ID ile korunur;
> yayınladığın içerikten sen sorumlusun.

## Ne yapar

1. YouTube linkini (ya da yerel dosyayı) alır.
2. Heyecanlı anları bulur:
   - **highlights modu** (özet video): videonun kendi sahne kesimleri + ses
     enerjisi sıralamasıyla en "büyük" parçaları seçer.
   - **match modu** (tam maç): senin elle girdiğin gol dakikalarının etrafında
     ses zirvesiyle tam saniyeye hizalar.
3. Her anı 9:16 dikey kadraja kırpar (1080x1920).
4. faster-whisper ile altyazı üretip videoya gömer (varsayılan açık).
5. `manifest.json` + `review.txt` yazar — **yayından önce elle kontrol** için.

## Kurulum

```bash
pip install -r requirements.txt
```

- Sistemde `ffmpeg` varsa onu kullanır; yoksa `imageio-ffmpeg` gömülü
  binary'sine otomatik düşer (ayrı kurulum gerekmez).
- `faster-whisper` modeli ilk çalıştırmada otomatik iner. Altyazı istemiyorsan
  `--no-subtitles` ile bu adımı atla.

## Kullanım

Özet videodan en iyi 5 klip:

```bash
python -m macshorts --url "https://youtube.com/watch?v=..." --mode highlights --count 5 --label "Dünya Kupası"
```

Tam maçtan elle gol dakikalarıyla:

```bash
python -m macshorts --url "https://youtube.com/watch?v=..." --mode match --minutes "23,45+2,67"
```

Yerel dosyadan (indirmeyi atla — test için ideal):

```bash
python -m macshorts --file mac.mp4 --mode highlights --count 3
```

Altyazısız, hızlı:

```bash
python -m macshorts --file mac.mp4 --no-subtitles
```

## Seçenekler

| Bayrak | Açıklama |
|---|---|
| `--url` / `--file` | Kaynak (biri zorunlu) |
| `--mode` | `highlights` (özet) veya `match` (tam maç) |
| `--count` | highlights modunda klip sayısı (varsayılan 5) |
| `--minutes` | match modu gol dakikaları, örn `23,45+2,67` |
| `--out` | Çıktı klasörü (varsayılan `output/`) |
| `--no-subtitles` | Altyazıyı atla |
| `--whisper-model` | tiny/base/small/medium (varsayılan small) |
| `--lang` | Altyazı dili (boşsa otomatik) |
| `--scene-threshold` | Sahne tespiti eşiği 0-1 (varsayılan 0.35) |
| `--label` | Önerilen başlık öneki |

## YouTube yayını (yarı-otomatik)

Sistem videoyu **PRIVATE** yükler, başlık/açıklama/hashtag'i transkriptten
otomatik üretir. Son "Yayınla" kararı sende kalır (YouTube Studio'da). Telif/spam
güvenliği için public yayın varsayılan değildir.

### Bir kerelik kurulum

1. https://console.cloud.google.com → yeni proje oluştur.
2. "APIs & Services → Library" → **YouTube Data API v3** → Enable.
3. "APIs & Services → OAuth consent screen" → External → uygulama adı gir,
   test kullanıcısı olarak kendi Google hesabını ekle.
4. "Credentials → Create Credentials → OAuth client ID" → tip: **Desktop app**.
5. İnen JSON'u proje köküne `client_secret.json` olarak koy.

### Kullanım

```bash
# whole modu + private yükleme:
python -m macshorts --url "https://instagram.com/p/..." --mode whole --vertical --publish --label "Dünya Kupası"
```

İlk `--publish` çalıştırmasında tarayıcı açılır, izin verirsin; token
`youtube_token.json`'a kaydedilir (sonraki sefer giriş gerekmez).

### Önemli sınırlar

- **Yükleme kotası:** Varsayılan API kotası günde ~6 video. Daha fazlası için
  Google'dan kota artışı iste.
- **Doğrulanmamış uygulama:** App "testing" modundayken yüklenen videolar
  PRIVATE'a kilitlenebilir; public yapmak için Studio'dan elle değiştir ya da
  app doğrulaması yaptır.
- `--privacy unlisted` / `--privacy public` ile gizlilik değişir (public riskli).

## Çıktı

Her çalıştırma `output/run-<tarih>/` altına:
- `clip-NN.mp4` / `clip-NN-sub.mp4` — üretilen Shorts klipleri
- `clip-NN.srt` — altyazı (yan dosya)
- `manifest.json` — makine-okunur kayıt
- `review.txt` — yayın öncesi insan kontrol listesi

## Sınırlar (tasarım v1, bilinçli ertelenenler)

- Merkez kırpma (topu takip eden akıllı kırpma ertelendi → tasarım Approach B).
- Otomatik yapısal gol verisi yok (maç-saati ↔ video-zaman uyumsuzluğu nedeniyle).
- Otomatik yayın yok (spam/telif riski nedeniyle bilinçli).
