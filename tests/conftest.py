"""Shared pytest configuration.

Both selected experiments use only data synthesized by the SFK null model
(data/generate_synthetic.py); no market data is loaded, so no market_data
fixture or marker is required here.
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: a longer-running test that reproduces a paper-scale decay exponent "
        "(skipped by default with `-m \"not slow\"`).",
    )
