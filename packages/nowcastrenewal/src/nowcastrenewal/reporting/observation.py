# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The reporting triangle's observed counts.

Each cell's count is drawn around its expected reports with the
simulation's count family (`components.noise`), its $\phi$ by stratum
($\phi = 1 / \sqrt{\phi}^2$ from `obs_sqrt_phi`). For NB1 the simulator
scales $\sqrt{\phi}$ by $\sqrt{\hat\lambda \sum_d p_d^2}$, with
$\hat\lambda$ the median expected onset among those of at least 1 and
$p$ the reporting delay's pmf (`nb1_scale`).
The noise is on the expected count $\mu$ itself.

The counts are one numpyro site, `"{name}::counts"`, which takes observed
counts to condition on, so the same model could be fitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

from ..components.noise import count_noise
from .delays import report_delay_pmf

__all__ = ["ObservedReports", "nb1_scale"]


def nb1_scale(
    onsets: ArrayLike,
    delay_family: str,
    values: Mapping[str, Any],
    max_delay: int,
) -> tuple[float, float, float]:
    r"""Return $\hat\lambda$, $\sum_d p_d^2$ and the NB1 scale.

    Parameters
    ----------
    onsets : array_like
        Expected onsets (reference events), any shape.
    delay_family : str
        The reporting delay's family, or `"none"`
        (`delays.report_delay_pmf`).
    values : mapping
        The values.
    max_delay : int
        Delays in the window.

    Returns
    -------
    tuple of float
        $\hat\lambda$ (at least 1), $\sum_d p_d^2$ and
        $\sqrt{\hat\lambda \sum_d p_d^2}$.
    """
    x = np.asarray(onsets, dtype=float)
    big = x[x >= 1]
    lam = max(1.0, float(np.median(big))) if big.size else 1.0
    pmf = report_delay_pmf(delay_family, values, max_delay)
    s2 = float((pmf**2).sum())
    return lam, s2, float(np.sqrt(lam * s2))


class ObservedReports(RandomVariable):
    r"""The observed counts of a reporting triangle (module docstring).

    Parameters
    ----------
    name : str
        Name of the component; the counts are `"{name}::counts"`.
    family : str
        `"poisson"`, `"negbin"` or `"negbin1d"`.
    sqrt_phi : array_like
        $1 / \sqrt{\phi}$ by stratum (`obs_sqrt_phi`), `(G,)`.
    onsets : array_like
        Expected onsets, for the NB1 scale.
    delay_family : str
        The reporting delay's family, or `"none"`.
    values : mapping
        The values.
    max_delay : int
        Delays in the window.
    """

    def __init__(
        self,
        name: str,
        family: str,
        sqrt_phi: ArrayLike,
        onsets: ArrayLike,
        delay_family: str,
        values: Mapping[str, Any],
        max_delay: int,
    ) -> None:
        super().__init__(name=name)
        sqrt_phi = np.asarray(sqrt_phi, dtype=float)
        if family == "negbin1d":
            sqrt_phi = (
                sqrt_phi * nb1_scale(onsets, delay_family, values, max_delay)[2]
            )
        self.family = family
        self.phi = 1 / sqrt_phi**2

    def sample(
        self,
        log_expected: ArrayLike,
        counts: ArrayLike | None = None,
        **kwargs: object,
    ) -> Array:
        r"""Draw, or condition on, the counts.

        Parameters
        ----------
        log_expected : array_like
            Log expected reports, `(G, T, D)`; $-\infty$ where none.
        counts : array_like, optional
            Observed counts to condition on.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Counts, `(G, T, D)`; 0 where none are expected.
        """
        mu = jnp.exp(jnp.asarray(log_expected))
        on = mu > 0
        # Cells with none expected are drawn at a tiny mean and set to 0,
        # and left out of the likelihood.
        safe = jnp.where(on, mu, 1e-12)
        phi = jnp.asarray(self.phi)[:, None, None]
        noise = count_noise(self.family, phi)
        y = noise.sample(f"{self.name}::counts", safe, obs=counts, mask=on)
        return jnp.where(on, y, 0)
