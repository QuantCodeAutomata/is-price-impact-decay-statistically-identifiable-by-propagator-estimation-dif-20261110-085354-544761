"""Sample second-order statistics and Wiener-Hopf deconvolution.

Custom implementation of the propagator deconvolution estimation reviewed in
the paper's Eqs. (4a)-(4c): no library implements this inverse problem, so it
is coded here directly from the paper's equations.

Given a stationary pair of series (x_n, y_n), the time-average estimators

    Cxy(tau) = (1 / (N - tau)) sum_{n=0}^{N-tau-1} x_n y_{n+tau}

are the sample analogues of the ensemble averages E[x_0 y_tau] used in the
paper (Eq. 4a defines R(tau) := E[epsilon_0 (m_tau - m_0)] and
C(tau) := E[epsilon_0 epsilon_tau] in the same way). Given the measured
autocorrelation C(tau) and response increment S(tau), the kernel increment
K(tau) solves the discrete Wiener-Hopf equation of the first kind (Eq. 4c)

    sum_{tau'=0}^{N_max} C(|tau - tau'|) K(tau') = S(tau),  tau = 0, ..., N_max,

a symmetric Toeplitz linear system solved here with scipy.linalg.solve_toeplitz
(Levinson-Durbin recursion), and the propagator is recovered by the partial
sum G(tau) = sum_{tau'=0}^{tau-1} K(tau') (Eq. 4b).
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def sample_cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """Time-average cross-correlation Cxy(tau) = mean_n x_n y_{n+tau}.

    Args:
        x: First series, length N.
        y: Second series, length N.
        max_lag: Largest lag tau to compute (inclusive).

    Returns:
        Array of length max_lag + 1 with Cxy(0), ..., Cxy(max_lag).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    n = x.shape[0]
    if max_lag < 0 or max_lag >= n:
        raise ValueError("max_lag must satisfy 0 <= max_lag < len(x)")
    out = np.empty(max_lag + 1, dtype=np.float64)
    for tau in range(max_lag + 1):
        out[tau] = np.dot(x[: n - tau], y[tau:]) / (n - tau)
    return out


def sample_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Time-average autocorrelation C(tau) = mean_n x_n x_{n+tau}."""
    return sample_cross_correlation(x, x, max_lag)


def solve_wiener_hopf(autocorrelation: np.ndarray, response_increment: np.ndarray) -> np.ndarray:
    """Solve the discrete Wiener-Hopf equation (Eq. 4c) for the kernel K(tau).

    Args:
        autocorrelation: C(0), ..., C(N_max), the first column/row of the
            symmetric Toeplitz matrix T_{ij} = C(|i - j|).
        response_increment: S(0), ..., S(N_max), the right-hand side.

    Returns:
        The kernel increment K(0), ..., K(N_max) solving T K = S.
    """
    autocorrelation = np.asarray(autocorrelation, dtype=np.float64)
    response_increment = np.asarray(response_increment, dtype=np.float64)
    if autocorrelation.shape != response_increment.shape:
        raise ValueError("autocorrelation and response_increment must have the same length")
    kernel = linalg.solve_toeplitz((autocorrelation, autocorrelation), response_increment)
    return kernel


def propagator_from_kernel(kernel: np.ndarray) -> np.ndarray:
    """Recover the propagator G(tau) = sum_{tau'=0}^{tau-1} K(tau') (Eq. 4b).

    Returns an array of the same length as ``kernel`` with G(0) = 0.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    propagator = np.concatenate(([0.0], np.cumsum(kernel)[:-1]))
    return propagator
