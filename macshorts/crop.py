"""Akıllı (aksiyon-takipli) 9:16 kırpma yörüngesi — tasarım Approach B.

Sabit merkez kırpma geniş maç planında topu/aksiyonu sık sık kadraj dışı
bırakır. Burada klibin hareketini küçük gri karelerle örnekler, kareler-arası
farktan yatay aksiyon merkezini bulur ve zamanla onu takip eden yumuşatılmış bir
yörünge çıkarırız. clip.cut_vertical_smart bunu bir ffmpeg `crop` ifadesine
çevirir: pencere, topun/oyunun olduğu tarafa kayar.

Yalnızca numpy + ffmpeg kullanır (OpenCV gibi ek bağımlılık yok). Hareket
sinyali zayıfsa (durağan plan) merkeze düşer; bu yüzden güvenli varsayılan
davranış, kötü durumda eski merkez kırpmaya yakındır.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .ffmpeg_tools import FfmpegError, ffmpeg_path

# Örnekleme çözünürlüğü. Yükseklik aspect'i bozsa da YATAY merkez kesri (0..1)
# doğru kalır; o yüzden sabit küçük tuval kullanırız (hız + bilinen şekil).
_SAMPLE_W = 96
_SAMPLE_H = 54


def _sample_gray(src: Path, start: float, duration: float, fps: float) -> np.ndarray:
    """[start, start+duration] aralığını fps ile küçük gri karelere indir.

    Returns: (n, H, W) uint8 dizisi. Kare yoksa boş dizi.
    """
    args = [
        ffmpeg_path(),
        "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-an",
        "-vf", f"fps={fps:g},scale={_SAMPLE_W}:{_SAMPLE_H},format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-800:]
        raise FfmpegError(f"Hareket örneklemesi başarısız:\n{tail}")
    frame_bytes = _SAMPLE_W * _SAMPLE_H
    n = len(proc.stdout) // frame_bytes
    if n == 0:
        return np.empty((0, _SAMPLE_H, _SAMPLE_W), dtype=np.uint8)
    buf = np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8)
    return buf.reshape(n, _SAMPLE_H, _SAMPLE_W)


def motion_trajectory(
    src: Path,
    start: float,
    duration: float,
    *,
    fps: float = 4.0,
    smooth_s: float = 1.0,
    max_pan_per_s: float = 0.35,
) -> list[tuple[float, float]]:
    """Klip boyunca yatay aksiyon merkezi yörüngesi.

    Returns: [(t_sn, merkez_kesri)] — t klip başına göreli (0..duration),
    merkez_kesri 0=sol .. 1=sağ kenar. Boş liste = sinyal yok (çağıran merkez
    kırpmaya düşmeli).

    Yöntem: ardışık karelerin mutlak farkı = hareket haritası. Sütun bazında
    toplanır -> yatay hareket profili; ağırlık merkezi = aksiyonun olduğu x.
    Sonra zaman ekseninde yumuşatılır (smooth_s) ve kayma hızı sınırlanır
    (max_pan_per_s) -> titreşimsiz, takip eden bir pencere.
    """
    frames = _sample_gray(src, start, duration, fps)
    if frames.shape[0] < 2:
        return []

    diff = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    col = diff.sum(axis=1).astype(np.float64)        # (n-1, W) yatay hareket profili
    col_idx = np.arange(_SAMPLE_W, dtype=np.float64)

    totals = col.sum(axis=1)                          # kare başına toplam hareket
    centroids = np.full(col.shape[0], 0.5)            # varsayılan: merkez
    active = totals > (totals.max() * 0.05 + 1e-9)    # önemsiz hareketi yok say
    centroids[active] = (col[active] @ col_idx) / totals[active] / (_SAMPLE_W - 1)

    # Hareketsiz kareleri en yakın aktif değere taşı (merkeze sıçramasın).
    if active.any():
        centroids = _fill_inactive(centroids, active)
    else:
        return []

    centroids = np.clip(centroids, 0.0, 1.0)

    # Zaman damgaları: i'inci fark ~ (i+1) ve i. karenin ortası.
    times = (np.arange(col.shape[0]) + 1.0) / fps

    centroids = _smooth(centroids, fps, smooth_s)
    centroids = _limit_pan(centroids, times, max_pan_per_s)

    # 0 ve duration uçlarını sabitle (ifade tüm aralığı kapsasın).
    pts: list[tuple[float, float]] = [(0.0, float(centroids[0]))]
    pts += [(float(t), float(c)) for t, c in zip(times, centroids)]
    pts.append((float(duration), float(centroids[-1])))
    return _downsample(pts, min_dt=0.4)


def _fill_inactive(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Aktif olmayan (hareketsiz) örnekleri en yakın aktif değerle doldur."""
    idx = np.where(active)[0]
    all_i = np.arange(len(values))
    nearest = idx[np.clip(np.searchsorted(idx, all_i), 0, len(idx) - 1)]
    # searchsorted sağ komşuyu verir; sol komşu daha yakınsa onu seç.
    left = idx[np.clip(np.searchsorted(idx, all_i) - 1, 0, len(idx) - 1)]
    pick_left = np.abs(all_i - left) < np.abs(all_i - nearest)
    chosen = np.where(pick_left, left, nearest)
    return values[chosen]


def _smooth(values: np.ndarray, fps: float, smooth_s: float) -> np.ndarray:
    """Hareketli ortalama ile yumuşat (kenar etkisini azaltacak şekilde)."""
    win = max(1, int(round(fps * smooth_s)))
    if win <= 1 or len(values) < 2:
        return values
    kernel = np.ones(win) / win
    padded = np.pad(values, (win, win), mode="edge")
    return np.convolve(padded, kernel, mode="same")[win:-win]


def _limit_pan(values: np.ndarray, times: np.ndarray, max_pan_per_s: float) -> np.ndarray:
    """Kayma hızını sınırla: pencere ani sıçramasın (yumuşak takip)."""
    out = values.copy()
    for i in range(1, len(out)):
        dt = max(times[i] - times[i - 1], 1e-3)
        max_step = max_pan_per_s * dt
        delta = out[i] - out[i - 1]
        if delta > max_step:
            out[i] = out[i - 1] + max_step
        elif delta < -max_step:
            out[i] = out[i - 1] - max_step
    return out


def _downsample(pts: list[tuple[float, float]], *, min_dt: float) -> list[tuple[float, float]]:
    """İfade boyutunu sınırlamak için ardışık çok yakın noktaları seyrelt.

    İlk ve son nokta hep korunur; aradakiler en az min_dt saniye aralıkla.
    """
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for t, c in pts[1:-1]:
        if t - out[-1][0] >= min_dt:
            out.append((t, c))
    out.append(pts[-1])
    return out


def crop_x_expr(traj: list[tuple[float, float]]) -> str:
    """Yörüngeyi ffmpeg `crop` x ifadesine çevir (parça-doğrusal, t'ye bağlı).

    Pencere genişliği cw = min(iw, ih*9/16). Merkez kesri cf(t) -> piksel:
        x = clamp(cf(t)*iw - cw/2, 0, iw-cw)
    cf(t) düz bir between()-toplamıdır (iç içe if yok): tam bir segment aktif
    olduğundan toplam doğru değeri verir. Tek-tırnakla sarılarak virgüller
    literal kalır (ffmpeg filtergraph kuralı), kaçış gerekmez.
    """
    cw = "min(iw,ih*9/16)"
    if not traj:
        cf = "0.5"
    elif len(traj) == 1:
        cf = f"{traj[0][1]:.4f}"
    else:
        terms = []
        for (t0, c0), (t1, c1) in zip(traj, traj[1:]):
            span = max(t1 - t0, 1e-3)
            # c0 + (c1-c0)*(t-t0)/span , YARI-AÇIK [t0,t1): knot'larda çift sayım
            # olmasın diye (between iki ucu da kapsardı).
            lin = f"({c0:.4f}+({c1 - c0:.4f})*(t-{t0:.3f})/{span:.3f})"
            terms.append(f"(gte(t,{t0:.3f})*lt(t,{t1:.3f}))*{lin}")
        # Son noktadan (ve tam knot t==t_last) sonrası için son değeri sabit tut.
        terms.append(f"gte(t,{traj[-1][0]:.3f})*{traj[-1][1]:.4f}")
        cf = "+".join(terms)
    return f"max(0,min(iw-{cw},({cf})*iw-({cw})/2))"
