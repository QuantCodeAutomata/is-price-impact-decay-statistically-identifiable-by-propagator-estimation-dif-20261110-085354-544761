"""Unit tests for the shared Wiener-Hopf deconvolution estimators."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from src.wiener_hopf import (
    propagator_from_kernel,
    sample_autocorrelation,
    sample_cross_correlation,
    solve_wiener_hopf,
)


def test_sample_autocorrelation_zero_lag_is_mean_square():
    x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    c = sample_autocorrelation(x, 3)
    assert c.shape == (4,)
    assert c[0] == pytest.approx(np.mean(x**2))


def test_sample_cross_correlation_matches_direct_computation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    max_lag = 10
    c = sample_cross_correlation(x, y, max_lag)
    for tau in range(max_lag + 1):
        expected = np.mean(x[: 200 - tau] * y[tau:])
        assert c[tau] == pytest.approx(expected)


def test_sample_cross_correlation_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        sample_cross_correlation(np.zeros(10), np.zeros(5), 2)


def test_sample_cross_correlation_rejects_invalid_max_lag():
    with pytest.raises(ValueError):
        sample_cross_correlation(np.zeros(10), np.zeros(10), 10)
    with pytest.raises(ValueError):
        sample_cross_correlation(np.zeros(10), np.zeros(10), -1)


def test_propagator_from_kernel_is_partial_sum_with_zero_at_origin():
    kernel = np.array([2.0, 3.0, -1.0, 4.0])
    g = propagator_from_kernel(kernel)
    assert g[0] == 0.0
    np.testing.assert_allclose(g, [0.0, 2.0, 5.0, 4.0])


def test_solve_wiener_hopf_recovers_known_kernel():
    """Build a known kernel K, synthesize S = Toeplitz(C) @ K for an arbitrary
    positive-definite autocorrelation C, and check the solver recovers K."""
    rng = np.random.default_rng(1)
    n = 40
    # A decaying, positive-definite autocorrelation (exponential kernel).
    tau = np.arange(n)
    autocorrelation = 0.9**tau
    true_kernel = rng.normal(size=n)
    toeplitz = linalg.toeplitz(autocorrelation)
    response = toeplitz @ true_kernel
    recovered_kernel = solve_wiener_hopf(autocorrelation, response)
    np.testing.assert_allclose(recovered_kernel, true_kernel, atol=1e-8)


def test_solve_wiener_hopf_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        solve_wiener_hopf(np.zeros(5), np.zeros(4))
