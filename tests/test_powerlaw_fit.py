"""Unit tests for the log-log power-law fitting helper (src/powerlaw_fit.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.powerlaw_fit import fit_power_law


def test_fit_recovers_known_exponent_and_amplitude():
    tau = np.arange(1, 2000, dtype=np.float64)
    amplitude, exponent = 2.5, -0.37
    y = amplitude * tau**exponent
    fit = fit_power_law(tau, y, tau_min=1, tau_max=1999)
    assert fit.exponent == pytest.approx(exponent, abs=1e-6)
    assert fit.amplitude == pytest.approx(amplitude, rel=1e-6)
    assert fit.r_value == pytest.approx(-1.0, abs=1e-6)


def test_fit_window_restricts_points_used():
    tau = np.arange(1, 100, dtype=np.float64)
    y = 1.0 * tau**-0.5
    fit = fit_power_law(tau, y, tau_min=10, tau_max=50)
    assert fit.tau_fit.min() >= 10
    assert fit.tau_fit.max() <= 50


def test_use_abs_handles_sign_changing_series():
    tau = np.arange(1, 200, dtype=np.float64)
    y = -3.0 * tau**-0.3  # always negative, as the inferred propagator can be
    fit = fit_power_law(tau, y, tau_min=1, tau_max=199, use_abs=True)
    assert fit.exponent == pytest.approx(-0.3, abs=1e-6)


def test_raises_when_too_few_positive_points():
    tau = np.array([1.0, 2.0])
    y = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        fit_power_law(tau, y, tau_min=1, tau_max=2)


def test_tau_zero_excluded_from_fit():
    tau = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1e6, 1.0, 2.0**-0.5, 3.0**-0.5, 4.0**-0.5])
    fit = fit_power_law(tau, y, tau_min=1, tau_max=4)
    assert 0.0 not in fit.tau_fit
