"""Akıllı kırpma ifade üretimi birim testleri (ffmpeg gerektirmez).

crop.crop_x_expr: yörüngeyi (zaman, merkez_kesri) ffmpeg crop x ifadesine
çevirir. Yörünge yardımcıları (yumuşatma, kayma sınırı) saf numpy.
"""
from __future__ import annotations

import numpy as np

from macshorts import crop


def test_empty_traj_is_center():
    expr = crop.crop_x_expr([])
    assert "0.5" in expr
    assert "min(iw,ih*9/16)" in expr


def test_single_point_constant():
    expr = crop.crop_x_expr([(0.0, 0.3)])
    assert "0.3" in expr
    assert "between" not in expr  # tek nokta -> sabit, segment yok


def test_multi_point_piecewise_no_double_count():
    # Yarı-açık segmentler: knot'ta tek segment aktif olmalı (gte*lt).
    traj = [(0.0, 0.2), (1.0, 0.8), (2.0, 0.5)]
    expr = crop.crop_x_expr(traj)
    assert "gte(t,0.000)*lt(t,1.000)" in expr
    assert "gte(t,1.000)*lt(t,2.000)" in expr
    # Son knot'tan sonrası sabit kuyruk.
    assert "gte(t,2.000)*0.5000" in expr


def test_expr_clamped_to_valid_range():
    expr = crop.crop_x_expr([(0.0, 0.1), (1.0, 0.9)])
    # x daima [0, iw-cw] arasına kıstırılır.
    assert expr.startswith("max(0,min(iw-min(iw,ih*9/16),")


def test_limit_pan_caps_velocity():
    vals = np.array([0.5, 1.0, 0.0])      # ani sıçramalar
    times = np.array([0.0, 1.0, 2.0])
    out = crop._limit_pan(vals, times, max_pan_per_s=0.2)
    assert abs(out[1] - out[0]) <= 0.2 + 1e-9
    assert abs(out[2] - out[1]) <= 0.2 + 1e-9


def test_smooth_reduces_variance():
    vals = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    out = crop._smooth(vals, fps=4.0, smooth_s=1.0)
    assert out.std() < vals.std()
    assert len(out) == len(vals)


def test_fill_inactive_no_center_jump():
    vals = np.array([0.5, 0.8, 0.5, 0.5])
    active = np.array([False, True, False, False])
    out = crop._fill_inactive(vals, active)
    # Hareketsiz kareler en yakın aktif (0.8) değerini almalı, 0.5'te kalmamalı.
    assert np.allclose(out, 0.8)
