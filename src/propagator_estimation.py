"""Linear propagator deconvolution estimation (paper Sec. "Propagator
deconvolution estimation", Eqs. 4a-4c; applied to the SFK model in Sec.
"SFK model as a null model" and Fig. 2).

Assembles the order-sign autocorrelation C(tau), the response increment
S(tau) and the inferred propagator G(tau) from one simulated SFK sample
path, using the shared estimators in src/wiener_hopf.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from src.sfk_model import SFKSimulation
from src.wiener_hopf import (
    propagator_from_kernel,
    sample_autocorrelation,
    sample_cross_correlation,
    solve_wiener_hopf,
)


@dataclass
class LinearPropagatorResult:
    """Output of the linear propagator deconvolution for one SFK sample.

    Attributes:
        lags: Lag values tau = 0, ..., max_lag.
        autocorrelation: Estimated order-sign autocorrelation C(tau)
            (Eq. 4a: C(tau) := E[epsilon_0 epsilon_tau]).
        response_increment: Estimated response increment S(tau)
            (Eq. 4a: S(tau) := R(tau + 1) - R(tau)).
        response_function: Estimated response function R(tau)
            (Eq. 4a: R(tau) := E[epsilon_0 (m_tau - m_0)]).
        kernel_increment: Inferred kernel increment K(tau) solving the
            discrete Wiener-Hopf equation (Eq. 4c).
        propagator: Inferred propagator G(tau) = sum_{tau'<tau} K(tau')
            (Eq. 4b).
    """

    lags: np.ndarray
    autocorrelation: np.ndarray
    response_increment: np.ndarray
    response_function: np.ndarray
    kernel_increment: np.ndarray
    propagator: np.ndarray


def estimate_linear_propagator(
    simulation: SFKSimulation, max_lag: int, wh_truncation: int | None = None
) -> LinearPropagatorResult:
    """Estimate C(tau), S(tau) and the inferred propagator G(tau).

    Args:
        simulation: One simulated SFK sample path.
        max_lag: Largest lag tau at which C(tau) and S(tau) are estimated.
        wh_truncation: Truncation N_max of the Wiener-Hopf sum (Eq. 4c); the
            paper does not state this value explicitly, so by default the
            full estimated range (wh_truncation = max_lag) is used, i.e. the
            sum is truncated exactly where the sample statistics stop being
            estimated.

    Returns:
        A LinearPropagatorResult with C(tau), S(tau), R(tau), K(tau), G(tau).
    """
    if wh_truncation is None:
        wh_truncation = max_lag
    if wh_truncation > max_lag:
        raise ValueError("wh_truncation cannot exceed max_lag")

    signs = simulation.signs.astype(np.float64)
    increments = simulation.increments

    autocorrelation = sample_autocorrelation(signs, max_lag)
    # S(tau) = E[epsilon_0 Delta m_tau] = R(tau + 1) - R(tau) by Eq. (4a),
    # obtained directly as the sign/increment cross-correlation (avoids
    # differencing noisy partial sums of the non-stationary price path).
    response_increment = sample_cross_correlation(signs, increments, max_lag)
    response_function = np.concatenate(([0.0], np.cumsum(response_increment)[:-1]))

    kernel_increment_trunc = solve_wiener_hopf(
        autocorrelation[: wh_truncation + 1], response_increment[: wh_truncation + 1]
    )
    kernel_increment = np.full(max_lag + 1, np.nan)
    kernel_increment[: wh_truncation + 1] = kernel_increment_trunc
    propagator_trunc = propagator_from_kernel(kernel_increment_trunc)
    propagator = np.full(max_lag + 1, np.nan)
    propagator[: wh_truncation + 1] = propagator_trunc

    return LinearPropagatorResult(
        lags=np.arange(max_lag + 1),
        autocorrelation=autocorrelation,
        response_increment=response_increment,
        response_function=response_function,
        kernel_increment=kernel_increment,
        propagator=propagator,
    )


def estimate_linear_propagator_mc_streaming(
    simulations: Iterable[SFKSimulation], max_lag: int, wh_truncation: int | None = None
) -> LinearPropagatorResult:
    """Streaming Monte Carlo version of estimate_linear_propagator_mc.

    Consumes ``simulations`` one sample path at a time (e.g. from
    data.generate_synthetic.iter_sfk_samples), accumulating the running sums
    needed for the across-path average of C(tau) and S(tau) in Eq. (4a)
    without holding more than one sample path in memory simultaneously. This
    is numerically identical to estimate_linear_propagator_mc applied to the
    materialized list, since both average the per-path estimators before the
    single Wiener-Hopf solve.

    Args:
        simulations: Iterable (list or generator) of independent SFK sample
            paths (same alpha, delta).
        max_lag: Largest lag tau at which C(tau) and S(tau) are estimated.
        wh_truncation: Truncation N_max of the Wiener-Hopf sum; see
            estimate_linear_propagator.

    Returns:
        A LinearPropagatorResult built from the across-path average of C, S.
    """
    if wh_truncation is None:
        wh_truncation = max_lag
    if wh_truncation > max_lag:
        raise ValueError("wh_truncation cannot exceed max_lag")

    autocorrelation_sum = np.zeros(max_lag + 1)
    response_increment_sum = np.zeros(max_lag + 1)
    n_paths = 0
    for sim in simulations:
        signs = sim.signs.astype(np.float64)
        autocorrelation_sum += sample_autocorrelation(signs, max_lag)
        response_increment_sum += sample_cross_correlation(signs, sim.increments, max_lag)
        n_paths += 1
    if n_paths == 0:
        raise ValueError("simulations must be a non-empty iterable")

    autocorrelation = autocorrelation_sum / n_paths
    response_increment = response_increment_sum / n_paths
    response_function = np.concatenate(([0.0], np.cumsum(response_increment)[:-1]))

    kernel_increment_trunc = solve_wiener_hopf(
        autocorrelation[: wh_truncation + 1], response_increment[: wh_truncation + 1]
    )
    kernel_increment = np.full(max_lag + 1, np.nan)
    kernel_increment[: wh_truncation + 1] = kernel_increment_trunc
    propagator_trunc = propagator_from_kernel(kernel_increment_trunc)
    propagator = np.full(max_lag + 1, np.nan)
    propagator[: wh_truncation + 1] = propagator_trunc

    return LinearPropagatorResult(
        lags=np.arange(max_lag + 1),
        autocorrelation=autocorrelation,
        response_increment=response_increment,
        response_function=response_function,
        kernel_increment=kernel_increment,
        propagator=propagator,
    )


def estimate_linear_propagator_mc(
    simulations: list[SFKSimulation], max_lag: int, wh_truncation: int | None = None
) -> LinearPropagatorResult:
    """Monte Carlo version: average C(tau) and S(tau) over sample paths, then
    deconvolve once from the averaged statistics.

    Averaging the two time-average estimators of Eq. (4a) across independent
    sample paths before solving the single Wiener-Hopf system (Eq. 4c) reduces
    estimation noise relative to deconvolving each path separately and
    averaging the resulting G(tau), since the deconvolution is a linear
    operator applied once to a less noisy input.

    Args:
        simulations: Independent SFK sample paths (same alpha, delta).
        max_lag: Largest lag tau at which C(tau) and S(tau) are estimated.
        wh_truncation: Truncation N_max of the Wiener-Hopf sum; see
            estimate_linear_propagator.

    Returns:
        A LinearPropagatorResult built from the across-path average of C, S.
    """
    if not simulations:
        raise ValueError("simulations must be a non-empty list")
    if wh_truncation is None:
        wh_truncation = max_lag
    if wh_truncation > max_lag:
        raise ValueError("wh_truncation cannot exceed max_lag")

    autocorrelations = np.stack(
        [sample_autocorrelation(sim.signs.astype(np.float64), max_lag) for sim in simulations]
    )
    response_increments = np.stack(
        [
            sample_cross_correlation(sim.signs.astype(np.float64), sim.increments, max_lag)
            for sim in simulations
        ]
    )
    autocorrelation = autocorrelations.mean(axis=0)
    response_increment = response_increments.mean(axis=0)
    response_function = np.concatenate(([0.0], np.cumsum(response_increment)[:-1]))

    kernel_increment_trunc = solve_wiener_hopf(
        autocorrelation[: wh_truncation + 1], response_increment[: wh_truncation + 1]
    )
    kernel_increment = np.full(max_lag + 1, np.nan)
    kernel_increment[: wh_truncation + 1] = kernel_increment_trunc
    propagator_trunc = propagator_from_kernel(kernel_increment_trunc)
    propagator = np.full(max_lag + 1, np.nan)
    propagator[: wh_truncation + 1] = propagator_trunc

    return LinearPropagatorResult(
        lags=np.arange(max_lag + 1),
        autocorrelation=autocorrelation,
        response_increment=response_increment,
        response_function=response_function,
        kernel_increment=kernel_increment,
        propagator=propagator,
    )
