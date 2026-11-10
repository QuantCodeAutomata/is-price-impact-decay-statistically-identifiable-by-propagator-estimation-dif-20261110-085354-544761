"""Exact SFK formulas for the M = 1 autocorrelation and response increment.

Closed-form expressions from the paper's End Matter / Appendix C, used only
to verify the simulation-based estimators of src/propagator_estimation.py on
a known answer (not used to generate the experiments' headline figures).

Exact autocorrelation (Eq. 400, with C_0 = 1):

    C_n = lambda/(zeta(alpha) Gamma(alpha+1))
          * int_0^inf dx x^alpha (1 - lambda + lambda e^{-x})^{n-1} / (e^x - 1)^2,  n >= 1

Exact response increment (Eq. 408-409):

    S_n = lambda*delta/(zeta(alpha) Gamma(alpha+1) Gamma(1-delta))
          * int_0^inf int_0^inf dx dy x^alpha y^{-1-delta} (1 - e^{-y})
            (1 - lambda + lambda e^{-(x+y)})^{n-1} / ((e^x - 1)(e^{x+y} - 1)),  n >= 1
    S_0 = zeta(alpha - delta + 1) / zeta(alpha)

Here lambda = 1/M is the per-step selection intensity of a splitter; the
paper's selected experiments use M = 1, i.e. lambda = 1.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import integrate, special

# Both integrands decay as exp(-x) (resp. exp(-(x+y))); beyond this cutoff the
# integrand underflows double precision (exp(-40) ~ 4e-18 relative to the
# integral's O(1) scale), so truncating here avoids overflow of x**alpha at
# the huge sample points scipy.integrate's infinite-interval transform probes,
# without changing the integral's value within its requested tolerance.
_INTEGRATION_CUTOFF = 40.0


def exact_autocorrelation(alpha: float, n_max: int, lam: float = 1.0) -> np.ndarray:
    """Exact sign-autocorrelation C_0, ..., C_{n_max} (Eq. 400)."""
    out = np.empty(n_max + 1, dtype=np.float64)
    out[0] = 1.0
    prefactor = lam / (special.zeta(alpha, 1) * special.gamma(alpha + 1.0))
    for n in range(1, n_max + 1):
        integrand = lambda x, n=n: (
            x**alpha * (1.0 - lam + lam * np.exp(-x)) ** (n - 1) / (np.expm1(x)) ** 2
        )
        # quad reports the tail near x -> 0 as "slowly convergent" for some
        # (alpha, n) even though the integrand x**(alpha-2) there is
        # integrable for alpha in (1, 2); the returned estimate is accurate
        # (see the mean-relative-error cross-check against the simulation
        # in exp/exp1_linear_propagator.py), so the benign warning is muted.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=integrate.IntegrationWarning)
            value, _ = integrate.quad(integrand, 0.0, _INTEGRATION_CUTOFF, limit=200)
        out[n] = prefactor * value
    return out


def exact_response_increment(alpha: float, delta: float, n_max: int, lam: float = 1.0) -> np.ndarray:
    """Exact response increment S_0, ..., S_{n_max} (Eqs. 408-409, 427)."""
    out = np.empty(n_max + 1, dtype=np.float64)
    out[0] = special.zeta(alpha - delta + 1.0, 1) / special.zeta(alpha, 1)
    prefactor = lam * delta / (
        special.zeta(alpha, 1) * special.gamma(alpha + 1.0) * special.gamma(1.0 - delta)
    )
    for n in range(1, n_max + 1):
        def integrand(y, x, n=n):
            return (
                x**alpha
                * y ** (-1.0 - delta)
                * (-np.expm1(-y))
                * (1.0 - lam + lam * np.exp(-(x + y))) ** (n - 1)
                / (np.expm1(x) * np.expm1(x + y))
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=integrate.IntegrationWarning)
            value, _ = integrate.dblquad(
                integrand,
                0.0,
                _INTEGRATION_CUTOFF,
                0.0,
                _INTEGRATION_CUTOFF,
                epsabs=1e-10,
                epsrel=1e-8,
            )
        out[n] = prefactor * value
    return out
