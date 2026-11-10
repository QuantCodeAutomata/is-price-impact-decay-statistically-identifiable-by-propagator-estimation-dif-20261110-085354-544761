"""Sato-Fujiwara-Kanazawa (SFK) order-splitting null model.

Implements the single-splitter (M = 1) dynamics of the SFK model as reviewed
in the paper's Part II, Sec. V "Model description" (Eqs. 397-398) and the
permanent square-root impact function of Eq. (10)/(399). With M = 1 the
splitter index i is always drawn, so Eqs. (397)-(398) reduce to a single
traders' metaorder sequence: a metaorder sign is drawn uniformly in
{-1, +1}, its size L is drawn from the power-law distribution
psi(L) = L^{-alpha-1} / zeta(alpha + 1), L >= 1 (Eq. 399), and L unit-volume
child orders of that sign are submitted one per simulation step. The price
increment at step n is

    Delta m_n = epsilon_n * Delta I(Q_n),   Delta I(Q) = I(Q + 1) - I(Q),

with I(Q) = Q^delta (Eq. 10) and Q_n the number of child orders of the
current metaorder already executed before step n. Because the contribution
of a completed metaorder is never revisited, the impact of every metaorder
is strictly permanent, as stated in Sec. V.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SFKSimulation:
    """Container for one simulated SFK sample path.

    Attributes:
        signs: Order sign epsilon_n of the child order submitted at step n,
            for n = 0, ..., n_steps - 1 (array of +-1, dtype int8).
        increments: Price increment Delta m_n = epsilon_n * Delta I(Q_n) for
            n = 0, ..., n_steps - 1 (float64 array).
        executed_volume: Executed volume Q_n of the active metaorder at step
            n, i.e. the number of child orders of that metaorder submitted
            strictly before step n (int64 array, 0-indexed as in Eq. 398b).
    """

    signs: np.ndarray
    increments: np.ndarray
    executed_volume: np.ndarray

    @property
    def price(self) -> np.ndarray:
        """Cumulative price path m_0, ..., m_{n_steps}, with m_0 = 0."""
        return np.concatenate(([0.0], np.cumsum(self.increments)))


def sample_metaorder_sizes(alpha: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw metaorder sizes L >= 1 from psi(L) = L^{-alpha-1}/zeta(alpha+1).

    This is exactly the probability mass function of the Zipf distribution,
    scipy.stats.zipf(a), with a = alpha + 1: f(k, a) = 1 / (zeta(a) k^a) for
    integer k >= 1 (SciPy ``zipf`` documentation). Paper Eq. (399).
    """
    if alpha <= 1.0 or alpha >= 2.0:
        raise ValueError(f"alpha must lie in (1, 2) per the LMF hypothesis, got {alpha}")
    return stats.zipf.rvs(alpha + 1.0, size=size, random_state=rng).astype(np.int64)


def simulate_sfk(
    alpha: float,
    delta: float,
    n_steps: int,
    rng: np.random.Generator,
    m_splitters: int = 1,
) -> SFKSimulation:
    """Simulate n_steps of the SFK model with a single metaorder splitter.

    Only M = 1 is implemented: the experiments reproduced here (Fig. 2 and
    Fig. 3 of the paper) both use M = 1 (single metaorder splitter), so the
    round-robin scheduling across M > 1 traders of Eq. (397)-(398) is not
    required here.

    Args:
        alpha: Metaorder-size distribution exponent, alpha in (1, 2).
        delta: Price-impact exponent, I(Q) = Q^delta, 0 < delta <= 1.
        n_steps: Number of simulation steps (child orders) to generate.
        rng: NumPy random generator used for all draws.
        m_splitters: Number of metaorder splitters M; only M = 1 supported.

    Returns:
        An SFKSimulation with the order-sign series, price increments and
        executed-volume series of length n_steps.
    """
    if m_splitters != 1:
        raise NotImplementedError(
            "Only M = 1 (single metaorder splitter) is implemented; "
            "both experiments reproduced here use M = 1."
        )
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if not (0.0 < delta <= 1.0):
        raise ValueError(f"delta must lie in (0, 1], got {delta}")

    # Generate metaorders (sign, size) in growing batches until their total
    # length covers n_steps, then flatten the ragged (sign, size) sequence
    # into per-step sign and executed-volume arrays with a single vectorized
    # searchsorted pass (no per-metaorder Python loop).
    signs_list: list[np.ndarray] = []
    sizes_list: list[np.ndarray] = []
    total = 0
    batch = max(1024, n_steps // 4)
    while total < n_steps:
        batch_signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=batch)
        batch_sizes = sample_metaorder_sizes(alpha, batch, rng)
        signs_list.append(batch_signs)
        sizes_list.append(batch_sizes)
        total += int(batch_sizes.sum())
        batch = max(1024, batch)

    signs_all = np.concatenate(signs_list)
    sizes_all = np.concatenate(sizes_list)
    cum = np.cumsum(sizes_all)
    n_metaorders = int(np.searchsorted(cum, n_steps, side="left") + 1)
    n_metaorders = min(n_metaorders, len(sizes_all))

    signs_all = signs_all[:n_metaorders]
    sizes_all = sizes_all[:n_metaorders]
    cum = cum[:n_metaorders]
    offsets = np.concatenate(([0], cum[:-1]))

    idx = np.arange(n_steps)
    group = np.searchsorted(cum, idx, side="right")
    executed_volume = (idx - offsets[group]).astype(np.int64)
    signs = signs_all[group].astype(np.int8)

    delta_impact = (executed_volume + 1.0) ** delta - executed_volume.astype(np.float64) ** delta
    increments = signs.astype(np.float64) * delta_impact

    return SFKSimulation(signs=signs, increments=increments, executed_volume=executed_volume)
