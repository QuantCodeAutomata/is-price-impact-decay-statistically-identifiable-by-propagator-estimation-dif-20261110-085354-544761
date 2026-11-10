"""Tests for the linear propagator deconvolution (src/propagator_estimation.py).

Checks the paper's qualitative predictions for the SFK null model (Sec. "SFK
model as a null model", Fig. 2): the inferred propagator G(tau) decays to
zero even though the true impact is permanent, and the decay exponent matches
beta_c = (1 - gamma) / 2 with gamma = alpha - 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import generate_sfk_sample, generate_sfk_samples
from src.powerlaw_fit import fit_power_law
from src.propagator_estimation import estimate_linear_propagator, estimate_linear_propagator_mc


@pytest.fixture(scope="module")
def small_simulation():
    return generate_sfk_sample(alpha=1.5, delta=0.5, n_steps=60_000, seed=123)


def test_autocorrelation_at_zero_lag_is_one(small_simulation):
    result = estimate_linear_propagator(small_simulation, max_lag=200)
    assert result.autocorrelation[0] == pytest.approx(1.0)


def test_response_function_is_cumulative_sum_of_increment(small_simulation):
    result = estimate_linear_propagator(small_simulation, max_lag=200)
    expected = np.concatenate(([0.0], np.cumsum(result.response_increment)[:-1]))
    np.testing.assert_allclose(result.response_function, expected)


def test_kernel_and_propagator_lengths_match_truncation(small_simulation):
    result = estimate_linear_propagator(small_simulation, max_lag=300, wh_truncation=150)
    assert result.lags.shape == (301,)
    assert np.isnan(result.propagator[151:]).all()
    assert not np.isnan(result.propagator[:151]).any()
    assert result.propagator[0] == 0.0


def test_wh_truncation_cannot_exceed_max_lag(small_simulation):
    with pytest.raises(ValueError):
        estimate_linear_propagator(small_simulation, max_lag=100, wh_truncation=200)


def test_mc_average_matches_manual_average_of_two_paths():
    sims = generate_sfk_samples(alpha=1.5, delta=0.5, n_steps=20_000, n_seeds=2, base_seed=7)
    mc_result = estimate_linear_propagator_mc(sims, max_lag=100)
    individual = [estimate_linear_propagator(sim, max_lag=100) for sim in sims]
    expected_autocorrelation = np.mean([r.autocorrelation for r in individual], axis=0)
    np.testing.assert_allclose(mc_result.autocorrelation, expected_autocorrelation)


def test_mc_requires_nonempty_list():
    with pytest.raises(ValueError):
        estimate_linear_propagator_mc([], max_lag=10)


@pytest.mark.slow
def test_inferred_propagator_decays_with_predicted_exponent():
    """Paper's universal decay theorem applied to the SFK null model: the
    inferred G(tau) ~ tau^{-beta_c} with beta_c = (1 - gamma) / 2,
    gamma = alpha - 1, for alpha = 1.5 (beta_c = 0.25), despite the true
    impact of the SFK model being strictly permanent (never decaying)."""
    alpha, delta = 1.5, 0.5
    max_lag = 1500
    sims = generate_sfk_samples(
        alpha=alpha, delta=delta, n_steps=2_000_000, n_seeds=4, base_seed=42
    )
    result = estimate_linear_propagator_mc(sims, max_lag=max_lag)

    gamma = alpha - 1.0
    beta_c = (1.0 - gamma) / 2.0

    # Autocorrelation decay exponent should be close to -gamma.
    c_fit = fit_power_law(result.lags, result.autocorrelation, tau_min=30, tau_max=800)
    assert c_fit.exponent == pytest.approx(-gamma, abs=0.25)

    # Inferred propagator should decay toward zero (transience), not plateau.
    g_fit = fit_power_law(result.lags, result.propagator, tau_min=30, tau_max=800)
    assert g_fit.exponent < 0
    assert g_fit.exponent == pytest.approx(-beta_c, abs=0.25)
    assert abs(result.propagator[max_lag]) < abs(result.propagator[30])
