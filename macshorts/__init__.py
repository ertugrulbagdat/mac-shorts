"""macshorts — YouTube maç videosundan 9:16 Shorts klipleri üreten hat."""

# Anaconda/MKL ile ctranslate2 (faster-whisper) arasındaki OpenMP runtime
# çakışmasını (OMP: Error #15) önle. numpy/ctranslate2 yüklenmeden ÖNCE
# ayarlanmalı, bu yüzden paket başında.
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Windows konsolu (cp1254 vb.) Türkçe/emoji karakterlerde print'i çökertebilir.
# stdout/stderr'i UTF-8'e sabitle ki başlık/açıklama yazdırırken patlama olmasın.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

__version__ = "0.1.0"
