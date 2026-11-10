"""Cross-check the simulation-based estimators against the SFK model's exact
closed-form formulas (src/exact_sfk_formulas.py), paper's End Matter / Appendix
C, Eqs. 400 and 408-409, for M = 1 (lambda = 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import generate_sfk_samples
from src.exact_sfk_formulas import exact_autocorrelation, exact_response_increment
from src.wiener_hopf import sample_autocorrelation, sample_cross_correlation


def test_exact_autocorrelation_c0_equals_one():
    c = exact_autocorrelation(alpha=1.5, n_max=5)
    assert c[0] == 1.0


def test_exact_autocorrelation_is_positive_and_decaying():
    c = exact_autocorrelation(alpha=1.5, n_max=30)
    assert np.all(c > 0.0)
    # Monotone decay is expected for this completely monotone kernel.
    assert np.all(np.diff(c) <= 1e-12)


def test_exact_response_increment_s0_matches_zeta_ratio():
    from scipy import special

    alpha, delta = 1.5, 0.5
    s0 = exact_response_increment(alpha, delta, n_max=1)[0]
    expected = special.zeta(alpha - delta + 1.0, 1) / special.zeta(alpha, 1)
    assert s0 == pytest.approx(expected)


@pytest.mark.slow
def test_simulated_autocorrelation_matches_exact_formula():
    """The time-average estimator of C(tau) over long SFK sample paths should
    agree with the exact integral formula (Eq. 400) within Monte Carlo noise."""
    alpha, delta = 1.5, 0.5
    n_max = 25
    sims = generate_sfk_samples(alpha=alpha, delta=delta, n_steps=400_000, n_seeds=6, base_seed=11)
    estimated = np.mean(
        [sample_autocorrelation(sim.signs.astype(np.float64), n_max) for sim in sims], axis=0
    )
    exact = exact_autocorrelation(alpha, n_max)
    # Loose tolerance: Monte Carlo noise at n_steps = 4e5 with 6 seeds.
    np.testing.assert_allclose(estimated, exact, atol=0.05)


@pytest.mark.slow
def test_simulated_response_increment_matches_exact_formula():
    alpha, delta = 1.5, 0.5
    n_max = 20
    sims = generate_sfk_samples(alpha=alpha, delta=delta, n_steps=400_000, n_seeds=6, base_seed=12)
    estimated = np.mean(
        [
            sample_cross_correlation(sim.signs.astype(np.float64), sim.increments, n_max)
            for sim in sims
        ],
        axis=0,
    )
    exact = exact_response_increment(alpha, delta, n_max)
    np.testing.assert_allclose(estimated, exact, atol=0.02)
