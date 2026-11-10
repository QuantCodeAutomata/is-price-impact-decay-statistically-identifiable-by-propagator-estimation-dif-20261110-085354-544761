"""Experiment 2: nonlinear propagator estimation with daily segmentation,
applied to the SFK null model.

Reproduces the paper's Fig. 3 (End Matter, "Nonlinear propagator estimation
and daily segmentation"): the SFK model (M = 1, alpha = 1.5, delta = 1/2) is
simulated once per sample path, then the daily-segmentation algorithm of
Ref. [18] partitions the path into segments of length D steps, builds the
signed square-root volume-imbalance proxy Psi_k^{(D)} (Eq. 19), and the
quasi-linear Wiener-Hopf deconvolution (Eq. 20) is applied at the segment
level for every D in {1, 3, 5, 10, 17, 31, 56, 100}.

The paper's point is that this artificial decay appears for every segment
length D, even though the SFK model has no true metaorder correlation beyond
a single completed metaorder: the segmentation algorithm itself manufactures
correlated proxies. Expected exponents (paper Sec. "Numerical demonstration"):

    C(tau) ~ tau^{-gamma},   gamma ~ alpha - 1 = 0.5
    S(tau) ~ tau^{delta - alpha} = tau^{-1}
    G(tau) ~ tau^{-beta_c},  beta_c = (1 - gamma) / 2 = 0.25

Outputs (plots, summary table, log) are written to results/.
"""

from __future__ import annotations

import json
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import iter_sfk_samples
from src import config
from src.nonlinear_propagator import estimate_nonlinear_propagator_mc_streaming
from src.powerlaw_fit import fit_power_law

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


def _plot_decay(
    tau: np.ndarray,
    y: np.ndarray,
    fit_exponent: float,
    fit_amplitude: float,
    theory_exponent: float,
    label: str,
    title: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    positive = (tau > 0) & (np.abs(y) > 0)
    ax.loglog(tau[positive], np.abs(y[positive]), ".", color="tab:blue", label=f"|{label}(tau)|")
    tail = tau[positive]
    ax.loglog(
        tail,
        fit_amplitude * tail**fit_exponent,
        "-",
        color="tab:red",
        label=f"fit: exponent={fit_exponent:.3f}",
    )
    ax.loglog(
        tail,
        np.abs(y[positive][0]) * (tail / tail[0]) ** theory_exponent,
        "--",
        color="black",
        label=f"theory: exponent={theory_exponent:.3f}",
    )
    ax.set_xlabel("tau = kD")
    ax.set_ylabel(f"|{label}(tau)|")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(n_steps: int, n_seeds: int) -> dict:
    """Run the nonlinear propagator estimation for every segment length D.

    Each D regenerates its own set of SFK sample paths through a fresh
    streaming generator (the SFK simulation itself is cheap to redraw, which
    keeps peak memory bounded to a single sample path at a time instead of
    holding one materialized list of paths per D).
    """
    alpha = config.EXP2_ALPHA
    gamma_theory = alpha - 1.0
    s_exponent_theory = config.IMPACT_EXPONENT_DELTA - alpha
    beta_c_theory = (1.0 - gamma_theory) / 2.0

    summary: dict = {}
    for d in config.EXP2_D_VALUES:
        max_lag_segments = max(config.EXP2_MIN_SEGMENT_LAGS, config.EXP2_MAX_LAG_TIME // d)
        n_segments = n_steps // d
        max_lag_segments = min(max_lag_segments, n_segments - 1)

        t0 = time.time()
        simulations = iter_sfk_samples(
            alpha=alpha,
            delta=config.IMPACT_EXPONENT_DELTA,
            n_steps=n_steps,
            n_seeds=n_seeds,
            base_seed=config.EXP2_RANDOM_SEED,
            m_splitters=config.M_SPLITTERS,
        )
        result = estimate_nonlinear_propagator_mc_streaming(
            simulations, segment_length=d, max_lag_segments=max_lag_segments
        )
        elapsed = time.time() - t0
        logger.info(
            "D=%d: %d segments, max_lag_segments=%d, max time lag=%d, in %.1fs",
            d, n_segments, max_lag_segments, max_lag_segments * d, elapsed,
        )

        fit_tau_max = min(config.EXP2_FIT_TAU_MAX, int(result.lags.max()) - d)
        fit_tau_max = max(fit_tau_max, config.EXP2_FIT_TAU_MIN + d)
        c_fit = fit_power_law(result.lags, result.autocorrelation, config.EXP2_FIT_TAU_MIN, fit_tau_max)
        s_fit = fit_power_law(result.lags, result.response_increment, config.EXP2_FIT_TAU_MIN, fit_tau_max)
        g_fit = fit_power_law(result.lags, result.propagator, config.EXP2_FIT_TAU_MIN, fit_tau_max)

        g_valid = ~np.isnan(result.propagator)
        g_infty_estimate = float(
            np.nanmean(result.propagator[g_valid][-max(1, max_lag_segments // 10) :])
        )

        for name, y, fit in (
            ("C", result.autocorrelation, c_fit),
            ("S", result.response_increment, s_fit),
            ("G", result.propagator, g_fit),
        ):
            theory = {"C": -gamma_theory, "S": s_exponent_theory, "G": -beta_c_theory}[name]
            _plot_decay(
                result.lags,
                y,
                fit.exponent,
                fit.amplitude,
                theory,
                name,
                f"D={d}: {name}(tau)",
                f"{RESULTS_DIR}/exp2_D{d}_{name}.png",
            )

        summary[str(d)] = {
            "segment_length_D": d,
            "n_segments": n_segments,
            "max_lag_segments": max_lag_segments,
            "gamma_theory": gamma_theory,
            "gamma_fit": -c_fit.exponent,
            "s_exponent_theory": s_exponent_theory,
            "s_exponent_fit": s_fit.exponent,
            "beta_c_theory": beta_c_theory,
            "beta_c_fit": -g_fit.exponent,
            "g_infty_estimate": g_infty_estimate,
            "c_fit_r2": c_fit.r_value**2,
            "s_fit_r2": s_fit.r_value**2,
            "g_fit_r2": g_fit.r_value**2,
            "n_steps": n_steps,
            "n_seeds": n_seeds,
            "elapsed_seconds": elapsed,
        }
        np.savez(
            f"{RESULTS_DIR}/exp2_D{d}_curves.npz",
            lags=result.lags,
            autocorrelation=result.autocorrelation,
            response_increment=result.response_increment,
            propagator=result.propagator,
        )

    return summary


def main(full: bool = False) -> dict:
    n_steps = config.EXP2_N_STEPS_PAPER if full else config.EXP2_N_STEPS_RUN
    n_seeds = config.EXP2_N_SEEDS_PAPER if full else config.EXP2_N_SEEDS_RUN
    logger.info(
        "exp2: realized configuration alpha=%.2f D_values=%s n_steps=%d n_seeds=%d",
        config.EXP2_ALPHA, config.EXP2_D_VALUES, n_steps, n_seeds,
    )
    summary = run_experiment(n_steps, n_seeds)
    with open(f"{RESULTS_DIR}/exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("exp2: summary written to %s/exp2_summary.json", RESULTS_DIR)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true", help="Use the paper-scale configuration explicitly"
    )
    args = parser.parse_args()
    main(full=args.full)
