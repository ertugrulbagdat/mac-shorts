"""SRT çeviri katmanı birim testleri (ağ/openai gerektirmez).

Kritik sözleşme: zaman damgaları, cue sayısı ve sırası ASLA değişmez; model
bir satırı atlarsa o cue İngilizce kalır.
"""
from __future__ import annotations

from types import SimpleNamespace

from macshorts import translate as tr

SAMPLE = """1
00:00:01,000 --> 00:00:02,500
What a goal!

2
00:00:02,600 --> 00:00:05,120
The keeper had no chance there.

3
00:00:05,500 --> 00:00:07,000
Offside, the flag is up.
"""


def _fake_client(handler):
    """handler(user_prompt) -> model yanıtı (metin)."""

    def create(*, model, messages, **kw):
        user = messages[-1]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=handler(user)))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _echo_tr(user: str) -> str:
    """Her satırı '[[n]] TR:<metin>' olarak döndüren sahte çevirmen."""
    out = []
    for line in user.splitlines():
        m = tr._TAG_RE.match(line)
        if m:
            out.append(f"[[{m.group(1)}]] TR:{m.group(2)}")
    return "\n".join(out)


def _patch(monkeypatch, handler):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setattr(tr, "_client", lambda api_key, base_url: _fake_client(handler))


def test_parse_render_roundtrip_preserves_timestamps():
    cues = tr.parse_srt(SAMPLE)
    assert [c.index for c in cues] == ["1", "2", "3"]
    assert cues[1].timing == "00:00:02,600 --> 00:00:05,120"
    assert cues[0].text == "What a goal!"
    assert tr.render_srt(cues).strip() == SAMPLE.strip()


def test_multiline_cue_flattened_but_timing_kept():
    cues = tr.parse_srt("7\n00:01:00,000 --> 00:01:02,000\nfirst line\nsecond line\n")
    assert len(cues) == 1
    assert cues[0].timing == "00:01:00,000 --> 00:01:02,000"
    assert cues[0].text == "first line second line"


def test_crlf_and_bom_tolerated():
    cues = tr.parse_srt("﻿1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\n")
    assert len(cues) == 1 and cues[0].text == "Hello"


def test_translate_cues_replaces_text_only(monkeypatch):
    _patch(monkeypatch, _echo_tr)
    cues = tr.parse_srt(SAMPLE)
    new, n = tr.translate_cues(cues, verbose=False)
    assert n == 3
    assert [c.timing for c in new] == [c.timing for c in cues]
    assert [c.index for c in new] == [c.index for c in cues]
    assert [c.text for c in new] == [
        "TR:What a goal!", "TR:The keeper had no chance there.",
        "TR:Offside, the flag is up.",
    ]


def test_skipped_line_keeps_original_text(monkeypatch):
    # Model 2. cue'yu hem grup hem tek-tek denemede atlıyor -> İngilizce kalmalı.
    def handler(user: str) -> str:
        out = []
        for line in user.splitlines():
            m = tr._TAG_RE.match(line)
            if m and m.group(2) != "The keeper had no chance there.":
                out.append(f"[[{m.group(1)}]] TR:{m.group(2)}")
        return "\n".join(out)

    _patch(monkeypatch, handler)
    new, n = tr.translate_cues(tr.parse_srt(SAMPLE), verbose=False)
    assert n == 2
    assert new[1].text == "The keeper had no chance there."
    assert new[1].timing == "00:00:02,600 --> 00:00:05,120"


def test_batch_misalignment_recovered_one_by_one(monkeypatch):
    # Grup isteğinde model saçmalıyor; tek cue'luk isteklerde doğru cevaplıyor.
    def handler(user: str) -> str:
        lines = user.splitlines()
        if len(lines) > 1:
            return "Elbette, işte çeviri:\nbir\niki\nüç"
        return _echo_tr(user)

    _patch(monkeypatch, handler)
    new, n = tr.translate_cues(tr.parse_srt(SAMPLE), verbose=False)
    assert n == 3
    assert all(c.text.startswith("TR:") for c in new)


def test_api_error_leaves_everything_original(monkeypatch):
    def boom(user: str) -> str:
        raise RuntimeError("502 Bad Gateway")

    monkeypatch.setattr(tr, "MAX_RETRIES", 1)
    _patch(monkeypatch, boom)
    cues = tr.parse_srt(SAMPLE)
    new, n = tr.translate_cues(cues, verbose=False)
    assert n == 0
    assert [c.text for c in new] == [c.text for c in cues]


def test_translate_srt_file_writes_translated_srt(monkeypatch, tmp_path):
    _patch(monkeypatch, _echo_tr)
    src = tmp_path / "clip-01.srt"
    src.write_text(SAMPLE, encoding="utf-8")
    dst = tmp_path / "clip-01.tr.srt"
    assert tr.translate_srt_file(src, dst, verbose=False) is True
    out = dst.read_text(encoding="utf-8")
    assert "00:00:02,600 --> 00:00:05,120" in out
    assert "TR:What a goal!" in out
    # Zaman damgası satır sayısı korunmuş olmalı
    assert out.count("-->") == SAMPLE.count("-->")


def test_missing_api_key_returns_false(monkeypatch, tmp_path):
    for name in tr.API_KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tr, "ENV_FILE", tmp_path / "yok.env")  # gerçek .env sızmasın
    src = tmp_path / "a.srt"
    src.write_text(SAMPLE, encoding="utf-8")
    dst = tmp_path / "a.tr.srt"
    assert tr.translate_srt_file(src, dst, verbose=False) is False
    assert not dst.exists()


def test_empty_srt_returns_false(tmp_path):
    src = tmp_path / "empty.srt"
    src.write_text("", encoding="utf-8")
    assert tr.translate_srt_file(src, tmp_path / "empty.tr.srt", verbose=False) is False


def test_api_key_loaded_from_dotenv(monkeypatch, tmp_path):
    for name in tr.API_KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("NVIDIA_API_KEY=nvapi-dosyadan", encoding="utf-8")
    monkeypatch.setattr(tr, "ENV_FILE", env)
    assert tr._api_key() == "nvapi-dosyadan"


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("NVIDIA_API_KEY=nvapi-dosyadan", encoding="utf-8")
    monkeypatch.setattr(tr, "ENV_FILE", env)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-ortamdan")
    assert tr._api_key() == "nvapi-ortamdan"
