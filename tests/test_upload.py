"""--upload: YouTube'a HER ZAMAN private yükleme (ağ / Google hesabı gerektirmez).

Google API istemcisi sahte modüllerle değiştirilir; böylece gerçek
videos.insert isteğinin gövdesi (privacyStatus) doğrulanır.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from macshorts import cli, pipeline
from macshorts import publish as pub
from macshorts.pipeline import ClipResult


# --- sahte Google API -------------------------------------------------------

@pytest.fixture
def fake_youtube(monkeypatch):
    """googleapiclient'ı taklit et; videos().insert(...) çağrılarını kaydet."""
    calls: list[dict] = []

    class _Request:
        def next_chunk(self):
            return None, {"id": f"vid{len(calls)}"}

    class _Videos:
        def insert(self, *, part, body, media_body):
            calls.append({"part": part, "body": body, "media": media_body})
            return _Request()

    class _Service:
        def videos(self):
            return _Videos()

    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = lambda name, ver, credentials: _Service()
    http = types.ModuleType("googleapiclient.http")
    http.MediaFileUpload = lambda path, chunksize, resumable: path
    monkeypatch.setitem(sys.modules, "googleapiclient", types.ModuleType("googleapiclient"))
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", discovery)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http)
    monkeypatch.setattr(pub, "_get_credentials", lambda cs, tok: object())
    return calls


META = {"title": "Başlık", "description": "Açıklama", "tags": ["a"]}


def test_upload_private_request_is_private(fake_youtube, tmp_path):
    url = pub.upload_private(tmp_path / "c.mp4", META,
                             client_secret=tmp_path / "cs.json", token_path=tmp_path / "t.json")
    [call] = fake_youtube
    assert call["body"]["status"]["privacyStatus"] == "private"
    assert call["body"]["snippet"]["title"] == "Başlık"
    assert call["part"] == "snippet,status"
    assert call["media"] == str(tmp_path / "c.mp4")
    assert url == "https://youtu.be/vid1"


def test_upload_private_has_no_privacy_parameter():
    with pytest.raises(TypeError):
        pub.upload_private(Path("c.mp4"), META, client_secret=Path("a"),
                           token_path=Path("b"), privacy="public")


def test_video_body_defaults_to_private():
    assert pub.video_body(META)["status"]["privacyStatus"] == "private"


# --- pipeline: kesim/çeviri sonrası yükleme ---------------------------------

def _result(tmp_path, idx=1, srt=None) -> ClipResult:
    return ClipResult(index=idx, file=str(tmp_path / f"clip-{idx:02d}-sub.mp4"),
                      start=0.0, end=20.0, peak=8.0, duration=20.0, score=1.0,
                      subtitled=bool(srt), srt=srt, suggested_title="x")


def test_upload_forces_private_even_if_privacy_public(fake_youtube, tmp_path):
    opts = pipeline.Options(source="x.mp4", upload=True, privacy="public")
    results = [_result(tmp_path, 1), _result(tmp_path, 2)]
    pipeline._maybe_publish(results, opts)

    assert len(fake_youtube) == 2
    assert all(c["body"]["status"]["privacyStatus"] == "private" for c in fake_youtube)
    assert [r.youtube_url for r in results] == ["https://youtu.be/vid1",
                                                "https://youtu.be/vid2"]


def test_no_upload_without_flag(fake_youtube, tmp_path):
    pipeline._maybe_publish([_result(tmp_path)], pipeline.Options(source="x.mp4"))
    assert fake_youtube == []


def test_publish_flag_still_honours_privacy(fake_youtube, tmp_path):
    opts = pipeline.Options(source="x.mp4", publish=True, privacy="unlisted")
    pipeline._maybe_publish([_result(tmp_path)], opts)
    assert fake_youtube[0]["body"]["status"]["privacyStatus"] == "unlisted"


def test_failed_upload_does_not_stop_others(monkeypatch, tmp_path, capsys):
    sent: list[str] = []

    def flaky(path, meta, **kw):
        if "clip-01" in str(path):
            raise RuntimeError("kota doldu")
        sent.append(str(path))
        return "https://youtu.be/ok"

    monkeypatch.setattr(pub, "upload_private", flaky)
    results = [_result(tmp_path, 1), _result(tmp_path, 2)]
    pipeline._maybe_publish(results, pipeline.Options(source="x.mp4", upload=True))

    assert results[0].youtube_url is None
    assert results[1].youtube_url == "https://youtu.be/ok"
    assert "kota doldu" in capsys.readouterr().out


def test_metadata_falls_back_to_file_name(monkeypatch, tmp_path):
    monkeypatch.setattr(pub, "build_metadata",
                        lambda **kw: (_ for _ in ()).throw(ValueError("bozuk srt")))
    meta = pipeline._upload_metadata(_result(tmp_path, 3), pipeline.Options(source="x"),
                                     None, part=3)
    assert meta["title"] == "clip-03-sub"
    assert "clip-03-sub" in meta["description"]


# --- CLI --------------------------------------------------------------------

def _capture_opts(monkeypatch) -> dict:
    box: dict = {}
    monkeypatch.setattr(cli, "run", lambda opts: box.update(opts=opts) or [object()])
    return box


def test_cli_upload_flag(monkeypatch):
    box = _capture_opts(monkeypatch)
    assert cli.main(["--file", "x.mp4", "--upload"]) == 0
    assert box["opts"].upload is True
    assert box["opts"].privacy == "private"


def test_cli_upload_off_by_default(monkeypatch):
    box = _capture_opts(monkeypatch)
    assert cli.main(["--file", "x.mp4"]) == 0
    assert box["opts"].upload is False


@pytest.mark.parametrize("privacy", ["public", "unlisted"])
def test_cli_upload_rejects_non_private(monkeypatch, privacy):
    _capture_opts(monkeypatch)
    assert cli.main(["--file", "x.mp4", "--upload", "--privacy", privacy]) == 2
