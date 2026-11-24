# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""Count observation noise: PyRenew's, plus the NB1 parameterisation.

`count_noise` gives the noise of a family: PyRenew's `PoissonNoise` and
`NegativeBinomialNoise` (NB2: variance $\mu + \mu^2 / \phi$), or
`NegativeBinomial1Noise` (NB1: variance $\mu (1 + 1 / \phi)$), which is
NB2 with concentration $\mu \phi$. PyRenew has no NB1, so it is added
here as a PyRenew `CountNoise`.
"""

from __future__ import annotations

import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable
from pyrenew.observation import NegativeBinomialNoise, PoissonNoise
from pyrenew.observation.noise import CountNoise

__all__ = ["FAMILIES", "NegativeBinomial1Noise", "count_noise"]

FAMILIES = ("poisson", "negbin", "negbin1d")
"""The count families: Poisson, NB2 and NB1."""


class NegativeBinomial1Noise(CountNoise):
    r"""NB1 noise: NB2 with concentration $\mu \phi$.

    The variance is $\mu (1 + 1 / \phi)$, linear in the mean.

    Parameters
    ----------
    phi_rv : RandomVariable
        Returns $\phi > 0$, broadcast against the predicted counts.
    """

    def __init__(self, phi_rv: RandomVariable) -> None:
        self.phi_rv = phi_rv

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NegativeBinomial1Noise(phi_rv={self.phi_rv!r})"

    def sample(
        self,
        name: str,
        predicted: ArrayLike,
        obs: ArrayLike | None = None,
        mask: ArrayLike | None = None,
    ) -> ArrayLike:
        """Sample, or condition on, counts with NB1 noise.

        Parameters
        ----------
        name : str
            Numpyro sample site name.
        predicted : array_like
            Predicted counts, positive.
        obs : array_like, optional
            Observed counts to condition on.
        mask : array_like, optional
            Which observations enter the likelihood.

        Returns
        -------
        ArrayLike
            The counts, the shape of `predicted`.
        """
        concentration = predicted * self.phi_rv()
        d = dist.NegativeBinomial2(mean=predicted, concentration=concentration)
        with numpyro.handlers.mask(mask=True if mask is None else mask):
            y: ArrayLike = numpyro.sample(name, d, obs=obs)
        return y


def count_noise(
    family: str, phi: ArrayLike | RandomVariable = 1.0
) -> CountNoise:
    r"""Return the noise of a count family.

    Parameters
    ----------
    family : str
        One of `FAMILIES`.
    phi : array_like or RandomVariable
        $\phi$ for the negative binomials (a fixed value is wrapped in a
        PyRenew `DeterministicVariable`); unused for the Poisson.

    Returns
    -------
    CountNoise
        PyRenew's `PoissonNoise` or `NegativeBinomialNoise`, or
        `NegativeBinomial1Noise`.

    Raises
    ------
    ValueError
        If `family` is not one of `FAMILIES`.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if family == "poisson":
        return PoissonNoise()
    phi_rv = (
        phi
        if isinstance(phi, RandomVariable)
        else DeterministicVariable(f"{family}_phi", phi)
    )
    if family == "negbin":
        return NegativeBinomialNoise(phi_rv)
    return NegativeBinomial1Noise(phi_rv)
