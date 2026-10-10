"""MRS 비중 규칙과 변동성 관리 비중 (사전등록 §6, §0-3.2)."""

from __future__ import annotations

import numpy as np

from modeler.scores.mrs import weights as W


def test_mrs_weight_boundaries_and_null():
    w = W.mrs_weight(np.array([0.0, 20.0, 35.0, 50.0, 65.0, 80.0, 100.0, np.nan]))
    np.testing.assert_allclose(w, [0.5, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0, 1.0])


def test_mrs_weight_just_inside_ramp():
    w = W.mrs_weight(np.array([20.0 + 1e-9, 80.0 - 1e-9]))
    assert 0.5 < w[0] < 0.5001 and 0.9999 < w[1] < 1.0


def test_vm_weight_clip_with_floor():
    s = np.array([0.01, 0.01, 0.01, 0.01])
    t = np.array([0.005, 0.0075, 0.01, 0.02])  # 비율 0.5, 0.75, 1.0, 2.0
    np.testing.assert_allclose(W.vm_weight(s, t), [0.5, 0.75, 1.0, 1.0])
    # 비율 0.2 -> 바닥 0.5
    assert W.vm_weight(np.array([0.05]), np.array([0.01]))[0] == 0.5


def test_vm_weight_nofloor_goes_below_half():
    s, t = np.array([0.05, 0.01]), np.array([0.01, 0.02])
    np.testing.assert_allclose(W.vm_weight(s, t, floor=False), [0.2, 1.0])
    np.testing.assert_allclose(W.vm_weight(s, t, floor=True), [0.5, 1.0])


def test_vm_weight_null_or_nonpositive_sigma_is_one():
    s = np.array([np.nan, 0.01, 0.0, 0.01])
    t = np.array([0.01, np.nan, 0.01, 0.005])
    np.testing.assert_allclose(W.vm_weight(s, t), [1.0, 1.0, 1.0, 0.5])
    np.testing.assert_allclose(W.vm_weight(s, t, floor=False), [1.0, 1.0, 1.0, 0.5])


def test_cap_is_min():
    np.testing.assert_allclose(W.cap(np.array([0.5, 0.7, 0.9, 1.0])), [0.5, 0.7, 0.7, 0.7])
    np.testing.assert_allclose(W.cap(np.array([0.5, 0.9]), 0.6), [0.5, 0.6])
