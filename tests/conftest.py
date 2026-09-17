"""Shared fixtures.

We size the reference corpus and the submission as the real campaign sized them
(507 trials against 40 model runs), for the sake of concreteness, so that a test
that turns on the ratio of the two turns on the ratio we actually face.
"""

import numpy as np
import pytest

REF_N = 507
MOD_N = 40
MU = 100.0
SIGMA = 10.0


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(7)


@pytest.fixture(scope="session")
def reference(rng):
    """A reference corpus at the size the evaluator ran."""
    return rng.normal(MU, SIGMA, REF_N)


@pytest.fixture(scope="session")
def zero_centred(rng):
    """A reference whose statistic is indistinguishable from zero."""
    return rng.normal(0.0, SIGMA, REF_N)


@pytest.fixture
def correct_model(rng):
    """A submission drawn from the reference's own distribution."""
    return rng.normal(MU, SIGMA, MOD_N)


@pytest.fixture
def fat_tailed_model(rng):
    """A right body with a wrong tail, namely the case a single central statistic
    misses.
    """
    return np.concatenate([rng.normal(MU, SIGMA, 34), rng.normal(170.0, SIGMA, 6)])
