"""Nonlinear (quasi-linear) propagator estimation with daily segmentation.

Implements the daily-segmentation algorithm and the quasi-linear propagator
model of Ref. [18] as reviewed in the paper's End Matter section "Nonlinear
propagator estimation and daily segmentation" (Eqs. 19-20). The market factor
and the normalizations of Ref. [18] are omitted, following the paper's
footnote 4, because the SFK model has a single asset with constant volume
and volatility.

Segmentation: the simulated step sequence is partitioned into contiguous
segments of D steps (the simulation analogue of one trading day). For
segment k, the metaorder volume imbalance is the net signed unit volume
submitted in that segment,

    Q_k^{(D)} = sum_{n=kD}^{(k+1)D-1} epsilon_n,

its sign is epsilon_k^{(D)} := sgn(Q_k^{(D)}), and the metaorder proxy is the
signed square-root volume imbalance Psi_k^{(D)} := epsilon_k^{(D)}
|Q_k^{(D)}|^{1/2} (Eq. 19). The D-step return is r_k^{(D)} := m_{(k+1)D} -
m_{kD}. The quasi-linear propagator model (Eq. 19) is

    r_k^{(D)} = sum_{l=0}^{inf} K_l^{(D)} Psi_{k-l}^{(D)} + xi_k^{(D)},

and the kernel is obtained from the autocorrelation C_k^{(D)} := E[Psi_l^{(D)}
Psi_{l-k}^{(D)}] and the response increment S_k^{(D)} := E[r_l^{(D)}
Psi_{l-k}^{(D)}] by the same Wiener-Hopf deconvolution as the linear case
(Eq. 20).
"""

from __future__ import annotations

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
class SegmentedSeries:
    """Daily-segmented metaorder proxies and D-step returns.

    Attributes:
        volume_imbalance: Q_k^{(D)}, net signed unit volume per segment.
        sign: epsilon_k^{(D)} = sgn(Q_k^{(D)}).
        signed_sqrt_imbalance: Psi_k^{(D)} = epsilon_k^{(D)} |Q_k^{(D)}|^{1/2}.
        segment_return: r_k^{(D)} = m_{(k+1)D} - m_{kD}.
    """

    volume_imbalance: np.ndarray
    sign: np.ndarray
    signed_sqrt_imbalance: np.ndarray
    segment_return: np.ndarray


def segment_metaorder_proxies(simulation: SFKSimulation, segment_length: int) -> SegmentedSeries:
    """Apply the daily-segmentation algorithm with segment length D.

    Args:
        simulation: One simulated SFK sample path.
        segment_length: Segment length D in simulation steps.

    Returns:
        A SegmentedSeries with K = n_steps // D complete segments.
    """
    if segment_length <= 0:
        raise ValueError("segment_length must be a positive integer")
    signs = simulation.signs.astype(np.float64)
    n_steps = signs.shape[0]
    n_segments = n_steps // segment_length
    usable = n_segments * segment_length
    signs = signs[:usable].reshape(n_segments, segment_length)

    volume_imbalance = signs.sum(axis=1)
    sign = np.sign(volume_imbalance)
    signed_sqrt_imbalance = sign * np.sqrt(np.abs(volume_imbalance))

    price = simulation.price  # length n_steps + 1, m_0, ..., m_{n_steps}
    sampled_price = price[0 : usable + 1 : segment_length]  # m_0, m_D, m_2D, ...
    segment_return = np.diff(sampled_price)  # r_k = m_{(k+1)D} - m_{kD}, length n_segments

    return SegmentedSeries(
        volume_imbalance=volume_imbalance,
        sign=sign,
        signed_sqrt_imbalance=signed_sqrt_imbalance,
        segment_return=segment_return,
    )


@dataclass
class NonlinearPropagatorResult:
    """Output of the nonlinear (quasi-linear) propagator deconvolution.

    Attributes:
        segment_lags: Segment-count lags k = 0, ..., max_lag_segments.
        lags: Time lags tau = k * D corresponding to segment_lags.
        autocorrelation: Estimated C_k^{(D)} = E[Psi_l Psi_{l-k}].
        response_increment: Estimated S_k^{(D)} = E[r_l Psi_{l-k}].
        kernel_increment: Inferred kernel increment K_k^{(D)} (Eq. 20).
        propagator: Inferred propagator G_k^{(D)} = sum_{l<k} K_l^{(D)}.
    """

    segment_lags: np.ndarray
    lags: np.ndarray
    autocorrelation: np.ndarray
    response_increment: np.ndarray
    kernel_increment: np.ndarray
    propagator: np.ndarray


def estimate_nonlinear_propagator(
    simulation: SFKSimulation,
    segment_length: int,
    max_lag_segments: int,
    wh_truncation: int | None = None,
) -> NonlinearPropagatorResult:
    """Estimate the quasi-linear propagator for one segment length D.

    Args:
        simulation: One simulated SFK sample path.
        segment_length: Segment length D (simulation steps per segment).
        max_lag_segments: Largest segment-count lag k at which C_k^{(D)} and
            S_k^{(D)} are estimated.
        wh_truncation: Truncation of the Wiener-Hopf sum in segment-count
            units; defaults to max_lag_segments (the paper notes that
            Ref. [18] truncates the sum at 50 days, but does not state
            whether the same truncation was used for the SFK demonstration).

    Returns:
        A NonlinearPropagatorResult with C_k^{(D)}, S_k^{(D)}, K_k^{(D)},
        G_k^{(D)} against both the segment-count lag k and the time lag
        tau = kD.
    """
    if wh_truncation is None:
        wh_truncation = max_lag_segments
    if wh_truncation > max_lag_segments:
        raise ValueError("wh_truncation cannot exceed max_lag_segments")

    segmented = segment_metaorder_proxies(simulation, segment_length)
    psi = segmented.signed_sqrt_imbalance
    returns = segmented.segment_return
    n_segments = psi.shape[0]
    if max_lag_segments >= n_segments:
        raise ValueError(
            f"max_lag_segments={max_lag_segments} must be < number of segments={n_segments}"
        )

    autocorrelation = sample_autocorrelation(psi, max_lag_segments)
    response_increment = sample_cross_correlation(returns, psi, max_lag_segments)

    kernel_increment_trunc = solve_wiener_hopf(
        autocorrelation[: wh_truncation + 1], response_increment[: wh_truncation + 1]
    )
    kernel_increment = np.full(max_lag_segments + 1, np.nan)
    kernel_increment[: wh_truncation + 1] = kernel_increment_trunc
    propagator_trunc = propagator_from_kernel(kernel_increment_trunc)
    propagator = np.full(max_lag_segments + 1, np.nan)
    propagator[: wh_truncation + 1] = propagator_trunc

    segment_lags = np.arange(max_lag_segments + 1)
    return NonlinearPropagatorResult(
        segment_lags=segment_lags,
        lags=segment_lags * segment_length,
        autocorrelation=autocorrelation,
        response_increment=response_increment,
        kernel_increment=kernel_increment,
        propagator=propagator,
    )


def estimate_nonlinear_propagator_mc(
    simulations: list[SFKSimulation],
    segment_length: int,
    max_lag_segments: int,
    wh_truncation: int | None = None,
) -> NonlinearPropagatorResult:
    """Monte Carlo version: average C_k^{(D)} and S_k^{(D)} over sample paths,
    then deconvolve once (see estimate_linear_propagator_mc for the rationale).
    """
    if not simulations:
        raise ValueError("simulations must be a non-empty list")
    if wh_truncation is None:
        wh_truncation = max_lag_segments
    if wh_truncation > max_lag_segments:
        raise ValueError("wh_truncation cannot exceed max_lag_segments")

    segmented_list = [segment_metaorder_proxies(sim, segment_length) for sim in simulations]
    n_segments = min(s.signed_sqrt_imbalance.shape[0] for s in segmented_list)
    if max_lag_segments >= n_segments:
        raise ValueError(
            f"max_lag_segments={max_lag_segments} must be < number of segments={n_segments}"
        )

    autocorrelations = np.stack(
        [sample_autocorrelation(s.signed_sqrt_imbalance, max_lag_segments) for s in segmented_list]
    )
    response_increments = np.stack(
        [
            sample_cross_correlation(s.segment_return, s.signed_sqrt_imbalance, max_lag_segments)
            for s in segmented_list
        ]
    )
    autocorrelation = autocorrelations.mean(axis=0)
    response_increment = response_increments.mean(axis=0)

    kernel_increment_trunc = solve_wiener_hopf(
        autocorrelation[: wh_truncation + 1], response_increment[: wh_truncation + 1]
    )
    kernel_increment = np.full(max_lag_segments + 1, np.nan)
    kernel_increment[: wh_truncation + 1] = kernel_increment_trunc
    propagator_trunc = propagator_from_kernel(kernel_increment_trunc)
    propagator = np.full(max_lag_segments + 1, np.nan)
    propagator[: wh_truncation + 1] = propagator_trunc

    segment_lags = np.arange(max_lag_segments + 1)
    return NonlinearPropagatorResult(
        segment_lags=segment_lags,
        lags=segment_lags * segment_length,
        autocorrelation=autocorrelation,
        response_increment=response_increment,
        kernel_increment=kernel_increment,
        propagator=propagator,
    )
