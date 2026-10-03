"""--duration / --horizontal testleri (ffmpeg gerektirmez)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from macshorts import cli, detect, pipeline
from macshorts.detect import Moment


# --- detect: klip süresi ---------------------------------------------------

def test_window_keeps_length_inside_video():
    assert detect._window(30.0, 8.0, 12.0, 100.0) == (22.0, 42.0)
    # başa yakın zirve: pencere sağa kayar, süre korunur
    assert detect._window(3.0, 24.0, 36.0, 100.0) == (0.0, 60.0)
    # sona yakın zirve: pencere sola kayar
    assert detect._window(98.0, 24.0, 36.0, 100.0) == (40.0, 100.0)
    # video klipten kısaysa tüm video
    assert detect._window(10.0, 24.0, 36.0, 30.0) == (0.0, 30.0)


def _fake_audio(monkeypatch, total: float, peaks: list[float]):
    sr = 1000
    samples = np.full(int(total * sr), 0.01, dtype=np.float32)
    for p in peaks:
        samples[int(p * sr):int((p + 1) * sr)] = 1.0
    monkeypatch.setattr(detect, "media_duration", lambda src: total)
    monkeypatch.setattr(detect, "extract_pcm", lambda src: (samples, sr))


def test_highlights_default_is_20s(monkeypatch):
    _fake_audio(monkeypatch, 300.0, [100.0])
    [m] = detect.detect_highlights(Path("x.mp4"), 1)
    assert m.duration == pytest.approx(20.0)


@pytest.mark.parametrize("clip_len", [20.0, 45.0, 60.0])
def test_highlights_respects_clip_len(monkeypatch, clip_len):
    _fake_audio(monkeypatch, 600.0, [100.0, 400.0])
    moments = detect.detect_highlights(Path("x.mp4"), 2, clip_len=clip_len)
    assert len(moments) == 2
    for m in moments:
        assert m.duration == pytest.approx(clip_len)
        assert m.start <= m.peak <= m.end


def test_match_respects_clip_len(monkeypatch):
    _fake_audio(monkeypatch, 3600.0, [1385.0])
    [m] = detect.detect_match(Path("x.mp4"), "23", clip_len=60.0)
    assert m.duration == pytest.approx(60.0)
    assert m.start <= m.peak <= m.end


# --- pipeline: kesim türü + süre aktarımı ----------------------------------

def _record_cuts(monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(pipeline.clipper, "cut_vertical",
                        lambda *a, **kw: calls.append("vertical"))
    monkeypatch.setattr(pipeline.clipper, "cut_segment",
                        lambda *a, **kw: calls.append("segment"))
    return calls


def _make(tmp_path, **kw):
    opts = pipeline.Options(source="x.mp4", subtitles=False, **kw)
    m = Moment(start=10.0, end=70.0, peak=34.0, score=1.0)
    return pipeline._make_clip(Path("x.mp4"), m, 1, tmp_path, opts)


def test_default_clip_is_vertical(monkeypatch, tmp_path):
    calls = _record_cuts(monkeypatch)
    _make(tmp_path)
    assert calls == ["vertical"]


def test_horizontal_keeps_aspect(monkeypatch, tmp_path):
    calls = _record_cuts(monkeypatch)
    res = _make(tmp_path, horizontal=True, smart_crop=True)
    assert calls == ["segment"]
    assert res.duration == 60.0


def test_detect_receives_duration(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline.detect, "detect_highlights",
                        lambda src, count, **kw: seen.update(kw) or [])
    pipeline._detect(Path("x.mp4"), pipeline.Options(source="x.mp4", duration=60.0))
    assert seen["clip_len"] == 60.0


# --- CLI -------------------------------------------------------------------

def _capture_opts(monkeypatch) -> dict:
    box: dict = {}

    def fake_run(opts):
        box["opts"] = opts
        return [object()]

    monkeypatch.setattr(cli, "run", fake_run)
    return box


def test_cli_defaults(monkeypatch):
    box = _capture_opts(monkeypatch)
    assert cli.main(["--file", "x.mp4"]) == 0
    assert box["opts"].duration is None
    assert box["opts"].horizontal is False


def test_cli_duration_and_horizontal(monkeypatch):
    box = _capture_opts(monkeypatch)
    assert cli.main(["--file", "x.mp4", "--duration", "60", "--horizontal"]) == 0
    assert box["opts"].duration == 60.0
    assert box["opts"].horizontal is True


@pytest.mark.parametrize("argv", [
    ["--file", "x.mp4", "--duration", "0"],
    ["--file", "x.mp4", "--duration", "-5"],
    ["--file", "x.mp4", "--horizontal", "--vertical"],
])
def test_cli_rejects_bad_combinations(monkeypatch, argv):
    _capture_opts(monkeypatch)
    assert cli.main(argv) == 2
