"""Tests for the daily-segmented nonlinear propagator estimation
(src/nonlinear_propagator.py), paper's End Matter, Eqs. 19-20.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import generate_sfk_sample
from src.nonlinear_propagator import (
    estimate_nonlinear_propagator,
    estimate_nonlinear_propagator_mc,
    segment_metaorder_proxies,
)


@pytest.fixture(scope="module")
def small_simulation():
    return generate_sfk_sample(alpha=1.5, delta=0.5, n_steps=60_000, seed=321)


def test_segment_count_and_shapes(small_simulation):
    segmented = segment_metaorder_proxies(small_simulation, segment_length=10)
    expected_segments = 60_000 // 10
    assert segmented.volume_imbalance.shape == (expected_segments,)
    assert segmented.sign.shape == (expected_segments,)
    assert segmented.signed_sqrt_imbalance.shape == (expected_segments,)
    assert segmented.segment_return.shape == (expected_segments,)


def test_signed_sqrt_imbalance_matches_definition(small_simulation):
    segmented = segment_metaorder_proxies(small_simulation, segment_length=5)
    expected = np.sign(segmented.volume_imbalance) * np.sqrt(np.abs(segmented.volume_imbalance))
    np.testing.assert_allclose(segmented.signed_sqrt_imbalance, expected)


def test_segment_length_one_reduces_to_per_step_series(small_simulation):
    """With D = 1, each segment is a single step: the imbalance sign equals
    the order sign and the return equals the price increment."""
    segmented = segment_metaorder_proxies(small_simulation, segment_length=1)
    np.testing.assert_allclose(segmented.sign, small_simulation.signs.astype(np.float64))
    np.testing.assert_allclose(segmented.segment_return, small_simulation.increments)


def test_segment_length_must_be_positive(small_simulation):
    with pytest.raises(ValueError):
        segment_metaorder_proxies(small_simulation, segment_length=0)


def test_estimate_nonlinear_propagator_output_shapes(small_simulation):
    result = estimate_nonlinear_propagator(small_simulation, segment_length=10, max_lag_segments=50)
    assert result.segment_lags.shape == (51,)
    np.testing.assert_array_equal(result.lags, result.segment_lags * 10)
    assert result.autocorrelation[0] == pytest.approx(
        np.mean(segment_metaorder_proxies(small_simulation, 10).signed_sqrt_imbalance ** 2)
    )


def test_max_lag_segments_must_be_less_than_segment_count(small_simulation):
    with pytest.raises(ValueError):
        estimate_nonlinear_propagator(small_simulation, segment_length=1000, max_lag_segments=1000)


def test_mc_requires_nonempty_list():
    with pytest.raises(ValueError):
        estimate_nonlinear_propagator_mc([], segment_length=5, max_lag_segments=10)


@pytest.mark.slow
@pytest.mark.parametrize("segment_length", [1, 10, 100])
def test_artificial_decay_persists_across_segment_lengths(segment_length):
    """Paper's key finding (End Matter, Fig. 3): the daily-segmentation
    algorithm produces an artificial transient, fine-tuned decay of the
    inferred propagator G(tau) for every segment length D, even though the
    SFK model's true metaorders have no cross-segment correlation structure
    beyond what segmentation itself introduces."""
    sim = generate_sfk_sample(alpha=1.5, delta=0.5, n_steps=1_000_000, seed=99)
    max_lag_segments = min(300, (sim.signs.shape[0] // segment_length) - 5)
    result = estimate_nonlinear_propagator(
        sim, segment_length=segment_length, max_lag_segments=max_lag_segments
    )
    tail = result.propagator[max(10, max_lag_segments // 4) :]
    head = result.propagator[1 : max(10, max_lag_segments // 4)]
    assert np.nanmean(np.abs(tail)) < np.nanmean(np.abs(head))
