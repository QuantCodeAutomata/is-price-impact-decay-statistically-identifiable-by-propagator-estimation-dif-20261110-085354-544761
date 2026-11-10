"""Power-law exponent fitting on log-log axes.

Custom log-log ordinary-least-squares fit (standard statistical tool, not a
dedicated power-law package): the paper reports decay laws of the form
f(tau) ~ A * tau^{-beta} for a deterministic second-order statistic f(tau)
(the sign autocorrelation C(tau), the response increment S(tau), and the
inferred propagator G(tau)), not a sample of i.i.d. draws from a heavy-tailed
distribution. Tail-index estimators for empirical distributions (such as the
``powerlaw`` package's Clauset-Shalizi-Newman MLE) answer a different
question and do not apply to a single decaying curve, so the exponent is
estimated here by ordinary least squares of log|f(tau)| on log(tau) over a
fitting window, using scipy.stats.linregress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PowerLawFit:
    """Result of a log-log power-law fit f(tau) ~ A * tau^{exponent}."""

    exponent: float
    amplitude: float
    r_value: float
    stderr: float
    tau_fit: np.ndarray
    y_fit: np.ndarray


def fit_power_law(
    tau: np.ndarray, y: np.ndarray, tau_min: int, tau_max: int, use_abs: bool = True
) -> PowerLawFit:
    """Fit f(tau) ~ A * tau^{exponent} over tau in [tau_min, tau_max].

    Args:
        tau: Lag values (>= 1 for the fit window; tau = 0 is never fit).
        y: Values of the statistic at each lag.
        tau_min: Smallest lag included in the fit window (inclusive).
        tau_max: Largest lag included in the fit window (inclusive).
        use_abs: Fit |y| instead of y (needed when y changes sign numerically
            near its asymptotic zero, e.g. the inferred propagator G(tau)).

    Returns:
        A PowerLawFit with the fitted exponent (the sign already included,
        so a decaying law has exponent < 0) and amplitude A = exp(intercept).
    """
    tau = np.asarray(tau, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (tau >= tau_min) & (tau <= tau_max) & (tau > 0)
    values = np.abs(y[mask]) if use_abs else y[mask]
    positive = values > 0
    if positive.sum() < 2:
        raise ValueError("Not enough strictly positive points in the fit window")
    log_tau = np.log(tau[mask][positive])
    log_y = np.log(values[positive])
    fit = stats.linregress(log_tau, log_y)
    return PowerLawFit(
        exponent=float(fit.slope),
        amplitude=float(np.exp(fit.intercept)),
        r_value=float(fit.rvalue),
        stderr=float(fit.stderr),
        tau_fit=tau[mask][positive],
        y_fit=values[positive],
    )
