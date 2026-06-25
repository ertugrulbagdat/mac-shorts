"""macshorts — YouTube maç videosundan 9:16 Shorts klipleri üreten hat."""

# Anaconda/MKL ile ctranslate2 (faster-whisper) arasındaki OpenMP runtime
# çakışmasını (OMP: Error #15) önle. numpy/ctranslate2 yüklenmeden ÖNCE
# ayarlanmalı, bu yüzden paket başında.
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
