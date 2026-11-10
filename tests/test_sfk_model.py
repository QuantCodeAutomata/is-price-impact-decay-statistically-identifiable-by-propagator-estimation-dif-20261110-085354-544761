"""Unit tests for the SFK null-model simulation (src/sfk_model.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.sfk_model import sample_metaorder_sizes, simulate_sfk


def test_signs_are_plus_minus_one():
    rng = np.random.default_rng(0)
    sim = simulate_sfk(alpha=1.5, delta=0.5, n_steps=5000, rng=rng)
    assert set(np.unique(sim.signs)) <= {-1, 1}
    assert sim.signs.dtype == np.int8


def test_output_lengths_match_n_steps():
    rng = np.random.default_rng(1)
    n_steps = 3000
    sim = simulate_sfk(alpha=1.5, delta=0.5, n_steps=n_steps, rng=rng)
    assert sim.signs.shape == (n_steps,)
    assert sim.increments.shape == (n_steps,)
    assert sim.executed_volume.shape == (n_steps,)
    assert sim.price.shape == (n_steps + 1,)


def test_executed_volume_resets_at_metaorder_boundaries():
    """Eq. (398b): executed volume resets to 0 whenever the sign changes
    (a new metaorder starts with a freshly drawn sign) or at the first step
    of a same-signed run that follows a completed metaorder."""
    rng = np.random.default_rng(2)
    sim = simulate_sfk(alpha=1.5, delta=0.5, n_steps=20000, rng=rng)
    # Every sign change must coincide with executed_volume resetting to 0.
    change = np.flatnonzero(np.diff(sim.signs) != 0) + 1
    assert np.all(sim.executed_volume[change] == 0)
    # Within a run of constant sign, executed_volume increases by exactly one
    # per step until the metaorder completes and the run ends or restarts.
    assert sim.executed_volume.min() == 0


def test_price_increment_consistent_with_impact_function():
    """Delta m_n = epsilon_n * (I(Q_n + 1) - I(Q_n)) with I(Q) = Q^delta (Eq. 10, 397)."""
    rng = np.random.default_rng(3)
    delta = 0.5
    sim = simulate_sfk(alpha=1.5, delta=delta, n_steps=10000, rng=rng)
    expected = sim.signs.astype(np.float64) * (
        (sim.executed_volume + 1.0) ** delta - sim.executed_volume.astype(np.float64) ** delta
    )
    np.testing.assert_allclose(sim.increments, expected, rtol=1e-12)


def test_price_is_permanent_cumulative_sum():
    rng = np.random.default_rng(4)
    sim = simulate_sfk(alpha=1.5, delta=0.5, n_steps=500, rng=rng)
    np.testing.assert_allclose(sim.price, np.concatenate(([0.0], np.cumsum(sim.increments))))


def test_metaorder_sizes_at_least_one_and_decay_with_alpha():
    rng = np.random.default_rng(5)
    sizes_small_alpha = sample_metaorder_sizes(1.25, 20000, rng)
    sizes_large_alpha = sample_metaorder_sizes(1.75, 20000, rng)
    assert sizes_small_alpha.min() >= 1
    assert sizes_large_alpha.min() >= 1
    # A smaller alpha (heavier tail) must produce a larger mean metaorder size.
    assert sizes_small_alpha.mean() > sizes_large_alpha.mean()


@pytest.mark.parametrize("alpha", [0.9, 2.0, 2.5])
def test_invalid_alpha_raises(alpha):
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        simulate_sfk(alpha=alpha, delta=0.5, n_steps=100, rng=rng)


def test_invalid_delta_raises():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        simulate_sfk(alpha=1.5, delta=0.0, n_steps=100, rng=rng)


def test_invalid_m_splitters_raises():
    rng = np.random.default_rng(8)
    with pytest.raises(NotImplementedError):
        simulate_sfk(alpha=1.5, delta=0.5, n_steps=100, rng=rng, m_splitters=2)


def test_non_positive_n_steps_raises():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError):
        simulate_sfk(alpha=1.5, delta=0.5, n_steps=0, rng=rng)


def test_sign_mean_close_to_zero_for_long_series():
    """Signs are +-1 with equal probability at each metaorder draw, so the
    long-run time-average sign should be close to zero."""
    rng = np.random.default_rng(10)
    sim = simulate_sfk(alpha=1.5, delta=0.5, n_steps=300000, rng=rng)
    assert abs(sim.signs.astype(np.float64).mean()) < 0.05
