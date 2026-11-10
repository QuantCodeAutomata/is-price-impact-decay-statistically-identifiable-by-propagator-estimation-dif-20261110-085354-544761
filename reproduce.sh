#!/usr/bin/env bash
# Runs the test suite and both experiments, exactly as in README.md.
set -uo pipefail

python -m pytest -m "not market_data"
python -m exp.exp1_linear_propagator
python -m exp.exp2_nonlinear_propagator
