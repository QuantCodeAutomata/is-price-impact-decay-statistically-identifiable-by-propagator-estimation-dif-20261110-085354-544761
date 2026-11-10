"""Experiment parameters for the SFK null-model propagator-estimation study.

All parameters are module-level constants, grouped by experiment. Each one
carries a comment stating its meaning, its unit, and its source: either a
paper equation/section, or "not specified in the paper" when the paper
leaves the value open (an uncertainty already flagged while retrieving the
methodology). No parameter is defined inside a class or a dict.
"""

# ---------------------------------------------------------------------------
# Shared SFK model parameters (paper Part II, Sec. V "Model description")
# ---------------------------------------------------------------------------
M_SPLITTERS = 1  # number of metaorder splitters M; both experiments use M = 1 (Figs. 2-3)
IMPACT_EXPONENT_DELTA = 0.5  # delta in I(Q) = Q^delta (Eq. 10); square-root law, delta = 1/2

# ---------------------------------------------------------------------------
# Experiment 1: linear propagator deconvolution (paper Fig. 2)
# ---------------------------------------------------------------------------
EXP1_ALPHAS = (1.25, 1.5, 1.75)  # metaorder-size exponents tested in Fig. 2

# Simulation length in steps (child orders). The paper's implementation notes
# state that 10^5-10^7 steps are needed for reliable large-lag estimation,
# without giving the exact value used for Fig. 2 (recorded uncertainty). The
# vectorized estimator keeps this length inexpensive, so the executed run
# uses the upper end of the suggested range directly (no reduction needed).
EXP1_N_STEPS_PAPER = 10_000_000  # upper end of the paper's suggested range
EXP1_N_STEPS_RUN = 10_000_000  # executed length; equals the paper range upper end

EXP1_N_SEEDS_PAPER = 20  # independent sample paths averaged; not stated in the paper
EXP1_N_SEEDS_RUN = 20  # executed number of independent sample paths

EXP1_MAX_LAG = 3000  # largest lag tau (steps) at which C(tau), S(tau), G(tau) are estimated
EXP1_WH_TRUNCATION = 3000  # Wiener-Hopf sum truncation N_max; not specified in the paper,
# set equal to EXP1_MAX_LAG so the deconvolution only uses lags that were actually estimated

EXP1_FIT_TAU_MIN = 50  # smallest lag included in the power-law tail fit (avoids short-lag transient)
EXP1_FIT_TAU_MAX = 2500  # largest lag included in the power-law tail fit

EXP1_EXACT_VERIFICATION_N_MAX = 60  # max lag for the exact-formula cross-check (Eqs. 400, 408-409)
EXP1_RANDOM_SEED = 20240601  # base seed for reproducibility

# ---------------------------------------------------------------------------
# Experiment 2: nonlinear propagator estimation with daily segmentation
# (paper Fig. 3, following Ref. [18])
# ---------------------------------------------------------------------------
EXP2_ALPHA = 1.5  # metaorder-size exponent used in Fig. 3
EXP2_D_VALUES = (1, 3, 5, 10, 17, 31, 56, 100)  # segment lengths D tested (steps per segment)

EXP2_N_STEPS_PAPER = 10_000_000  # paper's suggested upper simulation length (not stated exactly)
EXP2_N_STEPS_RUN = 10_000_000  # executed simulation length; equals the paper range upper end

EXP2_N_SEEDS_PAPER = 20  # independent sample paths averaged; not stated in the paper
EXP2_N_SEEDS_RUN = 20  # executed number of independent sample paths

EXP2_MAX_LAG_TIME = 3000  # largest real-time lag tau = kD at which statistics are estimated
# Segment-count lag cap per D: floor(EXP2_MAX_LAG_TIME / D), bounded below by 30 segments so
# that the largest D values still resolve a usable tail (both choices are not stated in the paper)
EXP2_MIN_SEGMENT_LAGS = 30

EXP2_WH_TRUNCATION_DAYS_PAPER = 50  # Ref. [18]'s own truncation at 50 days (footnote 4); the
# paper does not state whether the SFK demonstration reuses this value, so by default the
# deconvolution here truncates at the full estimated segment-lag range (see EXP2_MAX_LAG_TIME)

EXP2_FIT_TAU_MIN = 50  # smallest real-time lag tau = kD included in the power-law tail fit
EXP2_FIT_TAU_MAX = 2500  # largest real-time lag tau = kD included in the power-law tail fit

EXP2_RANDOM_SEED = 20240602  # base seed for reproducibility
