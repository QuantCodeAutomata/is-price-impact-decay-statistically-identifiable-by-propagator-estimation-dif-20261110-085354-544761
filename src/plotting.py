"""Log-log plotting of decay curves with power-law fit overlays.

Matplotlib is used directly (dedicated plotting library per the paper's
Fig. 2 and Fig. 3, which display C(tau), S(tau) and G(tau) on log-log axes).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.powerlaw_fit import PowerLawFit


def plot_decay_curve(
    tau: np.ndarray,
    y: np.ndarray,
    fit: PowerLawFit | None,
    theory_exponent: float,
    title: str,
    ylabel: str,
    out_path: str,
    use_abs: bool = True,
) -> None:
    """Plot one decay curve y(tau) on log-log axes with fit and theory lines.

    Args:
        tau: Lag values (tau = 0 is dropped before plotting on log axes).
        y: Statistic values at each lag.
        fit: Fitted PowerLawFit (or None to skip the fitted line).
        theory_exponent: Theoretical decay exponent (negative for decay),
            used to draw a reference line anchored at the fit window.
        title: Plot title.
        ylabel: Y-axis label.
        out_path: File path to save the figure (PNG).
        use_abs: Plot |y| (needed for G(tau), which can change sign near 0).
    """
    tau = np.asarray(tau, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (tau > 0) & np.isfinite(y)
    values = np.abs(y[mask]) if use_abs else y[mask]
    positive = values > 0

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(tau[mask][positive], values[positive], ".", markersize=3, label="estimated")
    if fit is not None:
        ax.loglog(
            fit.tau_fit,
            fit.amplitude * fit.tau_fit**fit.exponent,
            "-",
            color="tab:red",
            label=f"fit: exponent = {fit.exponent:.3f}",
        )
        anchor_tau = fit.tau_fit[0]
        anchor_y = fit.amplitude * anchor_tau**fit.exponent
        theory_amplitude = anchor_y / anchor_tau**theory_exponent
        ax.loglog(
            fit.tau_fit,
            theory_amplitude * fit.tau_fit**theory_exponent,
            "--",
            color="black",
            label=f"theory: exponent = {theory_exponent:.3f}",
        )
    ax.set_xlabel(r"lag $\tau$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
