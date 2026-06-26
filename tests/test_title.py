"""Başlık tespiti birim testleri (ffmpeg/ağ gerektirmez).

publish.build_metadata + yardımcıları: klibin transkriptinden anlamlı,
ayırt edici, gol/ofsayt/... farkındalıklı başlık üretimi.
"""
from __future__ import annotations

from macshorts import publish as pub


def test_garbled_rejected():
    assert pub._is_garbled("a a a th th uh uh mm mm")
    assert pub._is_garbled("aaaa")
    assert pub._is_garbled("...")
    assert not pub._is_garbled("The flag is up for offside.")


def test_event_keyword_word_boundary():
    # "gol" "golf"e takılmamalı; "var" (TR çok yaygın) olay sayılmamalı.
    assert pub._has_event("Penaltı verildi!")
    assert pub._has_event("he's in on goal!")
    assert not pub._has_event("Golf sahasında güzel bir gün.")
    assert not pub._has_event("VAR kararı bekleniyor.")


def test_short_event_sentence_wins_over_long_filler():
    # Kısa ama olaylı "Goal!" uzun ama olaysız cümleye yenmemeli.
    assert pub._best_headline("Looking to play this. It is in! Goal!") == "Goal!"


def test_offside_sentence_picked():
    txt = "The flag is up for offside. This one will not count."
    assert "offside" in pub._best_headline(txt).lower()


def test_junk_transcript_falls_back_to_caption():
    meta = pub.build_metadata(
        label="Dünya Kupası",
        srt_path=None,
        source={"caption": "Harika bir maç özeti.", "title": "Maç"},
        part=1,
    )
    # Transkript yok -> caption ilk cümlesinden başlık türemeli.
    assert "Harika bir maç" in meta["title"]


def test_title_within_youtube_limit():
    long_caption = "kelime " * 60
    meta = pub.build_metadata(
        label="Etiket", srt_path=None, source={"caption": long_caption}, part=None,
    )
    assert len(meta["title"]) <= 100
