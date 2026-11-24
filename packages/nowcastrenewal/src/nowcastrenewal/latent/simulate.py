# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A simulated outbreak's latent side, assembled from the components.

`simulate_latent` builds the outbreak's PyRenew components from a
simulation's design and values (layers 1 and 2 of the grid) and samples
them: strata weights, R by stratum (`rt.PlaceRt`), the start
(`start.InitializeStrata`), mixing, imports, the renewal
(`infections.StratifiedInfections`) and the expected reference events
(`components.reference.ReferenceEvents`). It is a numpyro model: run it under
`numpyro.handlers.seed` for a draw, or `numpyro.handlers.substitute` to
replay given draws. `simulate_latent_batch` runs a batch, one key each.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro import handlers
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import InfectionInitializationProcess
from pyrenew.math import get_asymptotic_growth_rate

from ..components.pmfs import Delay, delay_pmf, generation_time_pmf
from ..components.reference import ReferenceEvents
from ..components.renewal import DEPLETIONS, Shifts, Variants
from .imports import Imports
from .infections import StratifiedInfections
from .mixing import PlaceMixing
from .rt import KINDS, PlaceRt, RtPath, vector
from .start import InitializeStrata

__all__ = [
    "COLUMNS",
    "Latent",
    "round_up",
    "simulate_latent",
    "simulate_latent_batch",
]

COLUMNS = ("id", "day", "group", "infections", "onsets", "r", "reff")
"""The batch's long columns; group 0 is the whole outbreak, 1 to S the
strata in order."""


def _vector(values: Mapping[str, Any], name: str) -> np.ndarray:
    return vector(values.get(name))


@dataclass
class Latent:
    """A simulation's latent run.

    Attributes
    ----------
    onsets : numpy.ndarray
        Expected reference events (ascertained), days by strata.
    infections : numpy.ndarray
        Infections, days by strata, the generation time's span first.
    r, reff : numpy.ndarray
        Each stratum's R and effective R, NaN over the generation time's
        span, days by strata.
    path : pandas.DataFrame
        The whole outbreak by `day`: `infections`, `onsets`, and `r` and
        `reff` weighted by each day's infections (by population on days
        with none).
    stratum_population : numpy.ndarray
        Each stratum's population.
    attack : float
        Infections over the population of the strata with cases.
    n_gt : int
        Days in the generation time's span.
    """

    onsets: np.ndarray
    infections: np.ndarray
    r: np.ndarray
    reff: np.ndarray
    path: pd.DataFrame
    stratum_population: np.ndarray
    attack: float
    n_gt: int


def simulate_latent(
    n: int,
    gt: np.ndarray,
    strata: pd.DataFrame,
    design: Mapping[str, Any],
    values: Mapping[str, Any],
    kinds: Sequence[str] = KINDS,
    incubation_cv: float = 0.5,
    depletion: str = "linear",
    days: int | None = None,
    lag_bucket: int = 1,
) -> Latent:
    """Sample a simulation's latent run (a numpyro model).

    Strata weights are log-normal, quiet places have none, and each
    place's population is split over its ages by their weights. Late
    places, imports and a trickle into every stratum with cases come in
    as imports.

    Parameters
    ----------
    n : int
        Days after the generation time's span.
    gt : numpy.ndarray
        Generation time pmf (`components.pmfs.generation_time_pmf`).
    strata : pandas.DataFrame
        The strata, with `place` (`P01`, ...), `age` (`A1`, ...) and
        `.group` (1-based), in epinowcast's group order.
    design : mapping
        The simulation's design.
    values : mapping
        The simulation's values (layers 1 and 2).
    kinds : sequence of str
        The path kinds an independent place draws from (the grid's
        trajectory levels).
    incubation_cv : float
        The infection-to-reference delay's coefficient of variation when
        `values` has no `incubation_cv` (with a gamma when `design` has no
        `incubation_family`).
    depletion : str
        `"linear"` or `"exponential"` (`components.renewal.DEPLETIONS`).
    days : int, optional
        Days after the generation time's span that the arrays run over,
        at least `n`; `n` if None. Series of near lengths padded to the
        same days share compiled shapes. Timings refer to `n`, the run is
        causal and its days-first draws keep their values, so the run
        cropped to `n` days is the same.
    lag_bucket : int
        The generation time and the incubation pmfs are padded with zeros
        to a multiple of this many days, and the generation time's span
        with days before it that have no infections. The run cropped to
        the span is the same.

    Returns
    -------
    Latent
        The run over `n` days.

    Raises
    ------
    ValueError
        If the generation time is longer than the series, `days` is less
        than `n`, an established epidemic is larger than its population,
        or `depletion` is unknown.
    """
    if depletion not in DEPLETIONS:
        raise ValueError(f"depletion must be one of {tuple(DEPLETIONS)}")
    if n < 1:
        raise ValueError("the generation time is longer than the series")
    days = n if days is None else int(days)
    if days < n:
        raise ValueError("days must be at least n")
    v = values
    n_places, n_ages = int(design["n_places"]), int(design["n_ages"])
    place_ids = [f"P{i:02d}" for i in range(1, n_places + 1)]
    age_ids = [f"A{i}" for i in range(1, n_ages + 1)]
    sd = float(v["stratum_sd"])
    z_place = numpyro.sample(
        "place_weight", dist.Normal(0.0, 1.0).expand([n_places])
    )
    place_weight = np.exp(sd * np.asarray(z_place))
    quiet = _vector(v, "quiet_places").astype(int)
    place_weight[quiet - 1] = 0
    z_age = numpyro.sample("age_weight", dist.Normal(0.0, 1.0).expand([n_ages]))
    age_weight = np.exp(sd / 2 * np.asarray(z_age))
    at_place = np.array([place_ids.index(p) for p in strata["place"]])
    at_age = np.array([age_ids.index(a) for a in strata["age"]])
    weight = place_weight[at_place] * age_weight[at_age]
    active = weight > 0
    rel = weight / weight[active].mean()
    population = (
        _vector(v, "population")[at_place]
        * (age_weight / age_weight.sum())[at_age]
    )
    late_ids = {f"P{int(i):02d}" for i in _vector(v, "late_places")}
    late = np.asarray(
        (design["seeding"] == "mid") & strata["place"].isin(late_ids),
        dtype=bool,
    )
    on = active & ~late
    n_gt = len(gt)
    extra = round_up(n_gt, lag_bucket) - n_gt
    early = design["window"] == "early"
    unit = (v["seed"] if early else v["level"]) * rel
    path = RtPath.from_values(v["path"], v)
    # An outbreak that starts in the series grows into day 1 at the
    # growth of its R path's first day, so the renewal carries it on.
    growth = (
        float(get_asymptotic_growth_rate(np.exp(path.log_r(1, n)[0]), gt))
        if early
        else 1.0
    )
    start = InfectionInitializationProcess(
        "start",
        DeterministicVariable("start_unit", np.where(on, unit, 0.0)),
        InitializeStrata(n_gt + extra, early, active=n_gt, growth=growth),
    )
    # An established epidemic ran at its level before the span too: its
    # onsets take that history, about the level over the mean incubation.
    group = strata[".group"].to_numpy().astype(int) - 1
    history = (
        0.0
        if early
        else np.where(on, unit, 0.0) * _vector(v, "incubation_mean")[group]
    )
    if (np.asarray(start()).sum(axis=0) + history > population).any():
        raise ValueError("an established epidemic larger than its population")
    places = sorted(strata["place"].unique())
    place = strata["place"].map(places.index).to_numpy()
    age = strata["age"].to_numpy()
    process = StratifiedInfections(
        name="latent",
        gen_int_rv=DeterministicPMF("gt", np.pad(gt, (0, extra))),
        start_rv=start,
        log_rt_rv=PlaceRt(
            "rt",
            path=path,
            relation=design["place_rt"],
            kinds=kinds,
            values=v,
            noise_scale=float(v["gt_mean"]),
            place_sd=float(v.get("place_sd") or 0.0),
            place_rho=float(v.get("place_rho") or 0.0),
            place_region=_vector(v, "place_region").astype(int).tolist(),
            place_lags=_vector(v, "place_lags").astype(int).tolist(),
            drift=float(v["r_drift"]),
            coupling=float(v["coupling"]),
            horizon=n,
        ),
        population=population,
        place=place,
        age=age,
        mixing_rv=PlaceMixing(
            "mixing",
            kind=design["mixing"],
            share=float(v.get("mixing_eps") or 0.0),
            order=[int(i) - 1 for i in _vector(v, "mixing_order")],
            xy=_xy(v),
            distance_range=v.get("mixing_range"),
            quiet=[int(q) - 1 for q in quiet],
            switch=str(v.get("mixing_switch") or "on"),
            switch_at=float(v.get("mixing_switch_at") or 0.0),
            switch_days=float(v.get("mixing_switch_days") or 1.0),
            horizon=n,
        ),
        imports_rv=Imports(
            "imports",
            kind=design["imports"],
            rate=float(v.get("import_rate") or 0.0),
            pulse_at=_vector(v, "import_at"),
            pulse_size=float(v.get("import_size") or 0.0),
            spread=gt,
            late=late,
            late_at=float(v.get("late_at") or 0.0),
            late_days=int(v.get("late_days") or 1),
            trickle=float(v["trickle"]),
            horizon=n,
        ),
        unit=unit,
        shifts=Shifts(
            at=_vector(v, "shift_at") * n,
            size=_vector(v, "shift_size"),
            days=_vector(v, "shift_days"),
            reactive=_vector(v, "shift_reactive"),
            up=_vector(v, "shift_up"),
        ),
        variants=Variants(
            at=_vector(v, "variant_at") * n,
            escape=_vector(v, "variant_escape"),
            multiplier=_vector(v, "variant_r"),
            days=_vector(v, "variant_days"),
        ),
        depletion=DEPLETIONS[depletion],
        r_ceiling=(
            None if v.get("r_ceiling") is None else float(v["r_ceiling"])
        ),
    )
    with handlers.trace() as tr:
        sample = process(n_days_post_init=days)
    infections = np.asarray(sample.all_subpops)
    r = np.asarray(tr["latent::r"]["value"])[:n]
    reff = np.asarray(tr["latent::reff"]["value"])[:n]
    # The drawn infection-to-reference family and CV; a design that does
    # not draw them takes a gamma with `incubation_cv`.
    family = design.get("incubation_family", "gamma")
    cv = float(values.get("incubation_cv", incubation_cv))
    incubation = [
        delay_pmf(Delay.from_mean_cv(family, m, cv))
        for m in _vector(v, "incubation_mean")[group]
    ]
    width = round_up(max(len(p) for p in incubation), lag_bucket)
    events = ReferenceEvents(
        "reference",
        delay_rvs=[
            DeterministicPMF(f"incubation_{j}", np.pad(p, (0, width - len(p))))
            for j, p in enumerate(incubation)
        ],
        ascertainment_rv=DeterministicVariable(
            "ascertainment", _vector(v, "ascertainment")[group]
        ),
    )
    # An established or endemic stratum ran at its level before the span
    # too, the span's padding included; an outbreak that starts in it had
    # no infections before.
    before = None if early else np.asarray(start())[-1]
    history = infections.copy()
    if before is not None:
        history[:extra] = before
    onsets = np.asarray(events(infections=history, before=before))
    keep = slice(extra, extra + n_gt + n)
    return _latent(
        infections[keep],
        onsets[keep],
        r,
        reff,
        population,
        active,
        n_gt,
    )


def round_up(n: int, bucket: int) -> int:
    """Return `n` rounded up to a multiple of `bucket`.

    Parameters
    ----------
    n : int
        A length.
    bucket : int
        The multiple.

    Returns
    -------
    int
        The padded length.
    """
    return -(-int(n) // bucket) * bucket


def _xy(v: Mapping[str, Any]) -> np.ndarray | None:
    x, y = _vector(v, "mixing_x"), _vector(v, "mixing_y")
    if not len(x):
        return None
    m = min(len(x), len(y))
    return np.stack([x[:m], y[:m]], axis=1)


def _latent(
    infections: np.ndarray,
    onsets: np.ndarray,
    r: np.ndarray,
    reff: np.ndarray,
    population: np.ndarray,
    active: np.ndarray,
    n_gt: int,
) -> Latent:
    attack = float(infections[:, active].sum() / population[active].sum())
    inf = infections[n_gt:]
    total = inf.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = inf / total
    none = ~np.isfinite(w.sum(axis=1))
    w[none] = population / population.sum()
    pad = np.full(n_gt, np.nan)
    path = pd.DataFrame(
        {
            "day": np.arange(infections.shape[0]),
            "infections": infections.sum(axis=1),
            "onsets": onsets.sum(axis=1),
            "r": np.concatenate([pad, (r * w).sum(axis=1)]),
            "reff": np.concatenate([pad, (reff * w).sum(axis=1)]),
        }
    )
    pad2 = np.full((n_gt, infections.shape[1]), np.nan)
    return Latent(
        onsets=onsets,
        infections=infections,
        r=np.concatenate([pad2, r]),
        reff=np.concatenate([pad2, reff]),
        path=path,
        stratum_population=population,
        attack=attack,
        n_gt=n_gt,
    )


def _long(sim_id: str, latent: Latent) -> pd.DataFrame:
    days, k = latent.onsets.shape
    strata = pd.DataFrame(
        {
            "id": sim_id,
            "day": np.tile(np.arange(days), k),
            "group": np.repeat(np.arange(1, k + 1), days),
            "infections": latent.infections.T.ravel(),
            "onsets": latent.onsets.T.ravel(),
            "r": latent.r.T.ravel(),
            "reff": latent.reff.T.ravel(),
        }
    )
    whole = latent.path.assign(id=sim_id, group=0)
    return pd.concat([whole[list(COLUMNS)], strata], ignore_index=True)


def simulate_latent_batch(
    simulations: Sequence[Mapping[str, Any]],
    kinds: Sequence[str] = KINDS,
    incubation_cv: float = 0.5,
    depletion: str = "linear",
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run the latent side of a batch of simulations.

    Parameters
    ----------
    simulations : sequence of mapping
        Each with `id`, `key` (an integer, the run's JAX key), `n_days`,
        `design`, `values` and `strata` (records with `place`, `age` and
        `.group`).
    kinds : sequence of str
        The path kinds an independent place draws from.
    incubation_cv : float
        The infection-to-reference delay's coefficient of variation.
    depletion : str
        `"linear"` or `"exponential"`.

    Returns
    -------
    tuple
        The latent paths, long (`COLUMNS`), and each simulation's `n_gt`,
        generation time pmf `gt`, `stratum_population`, `attack` and
        `error` (None, or why it has no run: any error in one simulation
        is recorded, as the R batch redraws on any error).
    """
    frames: list[pd.DataFrame] = []
    out: dict[str, dict[str, Any]] = {}
    for sim in simulations:
        sim_id = str(sim["id"])
        design, values = sim["design"], sim["values"]
        try:
            gt = generation_time_pmf(
                Delay.from_mean_cv(
                    design["gt_family"], values["gt_mean"], values["gt_cv"]
                )
            )
            with jax.enable_x64(True):
                latent = handlers.seed(
                    simulate_latent, jax.random.key(int(sim["key"]))
                )(
                    int(sim["n_days"]) - len(gt),
                    gt,
                    pd.DataFrame(sim["strata"]),
                    design,
                    values,
                    kinds,
                    incubation_cv,
                    depletion,
                )
        except Exception as e:
            out[sim_id] = {
                "n_gt": None,
                "gt": None,
                "stratum_population": None,
                "attack": None,
                "error": str(e)
                if type(e) is ValueError
                else f"{type(e).__name__}: {e}",
            }
            continue
        frames.append(_long(sim_id, latent))
        out[sim_id] = {
            "n_gt": latent.n_gt,
            "gt": gt.tolist(),
            "stratum_population": latent.stratum_population.tolist(),
            "attack": latent.attack,
            "error": None,
        }
    long = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(COLUMNS))
    )
    return long, out
