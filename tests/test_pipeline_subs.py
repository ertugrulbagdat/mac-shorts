"""Altyazı zinciri kablolama testleri (ffmpeg/whisper/ağ gerektirmez).

Doğrulanan: whisper SRT -> NVIDIA çevirisi -> ffmpeg'e GÖMÜLEN dosya çevrilmiş
SRT olmalı; çeviri düşerse İngilizce SRT gömülmeli.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from macshorts import pipeline, subtitles
from macshorts import translate as tr

SRT_EN = "1\n00:00:01,000 --> 00:00:02,000\nWhat a goal!\n"


def _opts(**kw) -> pipeline.Options:
    return pipeline.Options(source="x.mp4", **kw)


def _stub_whisper(monkeypatch, ok: bool = True):
    def fake(clip, srt_path, model, lang):
        if not ok:
            return False
        Path(srt_path).write_text(SRT_EN, encoding="utf-8")
        return True

    monkeypatch.setattr(subtitles, "transcribe_to_srt", fake)


def _stub_burn(monkeypatch, burned: list, ok: bool = True):
    def fake(clip, srt_path, dst, *, font_size=12, margin_v=45):
        burned.append(Path(srt_path))
        if ok:
            Path(dst).write_text("fake mp4", encoding="utf-8")
        return ok

    monkeypatch.setattr(subtitles, "burn", fake)


def _stub_translate(monkeypatch, ok: bool = True):
    def fake(src, dst, **kw):
        if not ok:
            return False
        cues = tr.parse_srt(Path(src).read_text(encoding="utf-8"))
        for c in cues:
            c.text = "Ne gol!"
        Path(dst).write_text(tr.render_srt(cues), encoding="utf-8")
        return True

    monkeypatch.setattr(tr, "translate_srt_file", fake)


def test_translated_srt_is_the_one_burned(monkeypatch, tmp_path):
    burned: list[Path] = []
    _stub_whisper(monkeypatch)
    _stub_burn(monkeypatch, burned)
    _stub_translate(monkeypatch)

    media = tmp_path / "clip-01.mp4"
    media.write_text("fake", encoding="utf-8")
    res = pipeline._apply_subtitles(media, tmp_path / "clip-01", _opts())

    assert res.subtitled is True
    assert res.translated is True
    assert burned == [tmp_path / "clip-01.tr.srt"]      # ffmpeg'e giden dosya
    assert res.srt == str(tmp_path / "clip-01.tr.srt")
    assert res.srt_source == str(tmp_path / "clip-01.srt")
    assert "Ne gol!" in Path(res.srt).read_text(encoding="utf-8")
    assert res.final == tmp_path / "clip-01-sub.mp4"


def test_failed_translation_falls_back_to_english(monkeypatch, tmp_path):
    burned: list[Path] = []
    _stub_whisper(monkeypatch)
    _stub_burn(monkeypatch, burned)
    _stub_translate(monkeypatch, ok=False)

    media = tmp_path / "clip-01.mp4"
    media.write_text("fake", encoding="utf-8")
    res = pipeline._apply_subtitles(media, tmp_path / "clip-01", _opts())

    assert res.subtitled is True
    assert res.translated is False
    assert burned == [tmp_path / "clip-01.srt"]
    assert res.srt == str(tmp_path / "clip-01.srt")


def test_no_translate_option_skips_translation(monkeypatch, tmp_path):
    burned: list[Path] = []
    _stub_whisper(monkeypatch)
    _stub_burn(monkeypatch, burned)

    def must_not_run(*a, **kw):
        raise AssertionError("--no-translate iken çeviri çağrılmamalı")

    monkeypatch.setattr(tr, "translate_srt_file", must_not_run)

    media = tmp_path / "video.mp4"
    media.write_text("fake", encoding="utf-8")
    res = pipeline._apply_subtitles(media, tmp_path / "video", _opts(translate=False))

    assert burned == [tmp_path / "video.srt"]
    assert res.translated is False


def test_no_transcript_means_no_translation_call(monkeypatch, tmp_path):
    _stub_whisper(monkeypatch, ok=False)
    monkeypatch.setattr(subtitles, "burn", lambda *a, **kw: AssertionError())
    monkeypatch.setattr(
        tr, "translate_srt_file",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("çağrılmamalı")),
    )

    media = tmp_path / "clip-01.mp4"
    res = pipeline._apply_subtitles(media, tmp_path / "clip-01", _opts())
    assert res == pipeline._SubResult(final=media, subtitled=False)


def test_burn_failure_keeps_srt_as_sidecar(monkeypatch, tmp_path):
    burned: list[Path] = []
    _stub_whisper(monkeypatch)
    _stub_burn(monkeypatch, burned, ok=False)
    _stub_translate(monkeypatch)

    media = tmp_path / "clip-01.mp4"
    media.write_text("fake", encoding="utf-8")
    res = pipeline._apply_subtitles(media, tmp_path / "clip-01", _opts())

    assert res.subtitled is False
    assert res.final == media
    assert res.srt == str(tmp_path / "clip-01.tr.srt")   # yan dosya olarak kalır
    assert res.translated is True


def test_check_translate_ready_disables_without_api_key(monkeypatch, tmp_path):
    for name in tr.API_KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tr, "ENV_FILE", tmp_path / "yok.env")  # gerçek .env sızmasın
    opts = _opts()
    pipeline._check_translate_ready(opts)
    assert opts.translate is False


def test_check_translate_ready_keeps_enabled_with_api_key(monkeypatch):
    pytest.importorskip("openai")        # openai yoksa çeviri zaten kapatılır
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    opts = _opts()
    pipeline._check_translate_ready(opts)
    assert opts.translate is True
