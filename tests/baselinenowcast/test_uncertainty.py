"""Uncertainty estimation against R baselinenowcast 0.2.2."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from bnc_helpers import (
    NB_FITS,
    TOY,
    Case,
    first_delays,
    from_na,
    read,
    read_vec,
)

from nowcastbaselines.baselinenowcast import (
    estimate_uncertainty,
    estimate_uncertainty_retro,
    fit_by_horizon,
    fit_nb,
    preprocess_negative_values,
    rolling_sum,
)

MODES = ["id", "agg", "week"]


def _capture() -> tuple[dict[str, Any], Any]:
    store: dict[str, Any] = {}

    def model(obs: Any, pred: Any) -> Any:
        store["obs"] = np.asarray(obs)
        store["pred"] = np.asarray(pred)
        return fit_by_horizon(obs, pred)

    return store, model


def _retro(case: Case, **kwargs: Any) -> Any:
    pre = preprocess_negative_values(case.values, case.mask)
    return estimate_uncertainty_retro(
        pre, case.mask, case.n_hist, case.n_retro, **kwargs
    )


def _aggregators(case: Case, mode: str) -> dict[str, Any]:
    """The aggregators of R's mode (`bnc_modes` in r/cases.R)."""
    out: dict[str, Any] = {}
    if mode in ("agg", "week"):
        out["ref_time_aggregator"] = partial(rolling_sum, k=case.k)
    if mode == "week":
        out["delay_aggregator"] = first_delays
    return out


@pytest.mark.parametrize("mode", MODES)
def test_retro_obs_and_pred_match_r(case: Case, mode: str) -> None:
    # The totals the uncertainty model is fitted to, to 1e-10 relative
    # (1e-9 for the predictions, which go through the delay estimate).
    store, model = _capture()
    _retro(case, uncertainty_model=model, **_aggregators(case, mode))
    np.testing.assert_allclose(
        store["obs"],
        read(case.name, f"retro_obs_{mode}"),
        rtol=1e-10,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        store["pred"],
        read(case.name, f"retro_pred_{mode}"),
        rtol=1e-9,
        atol=1e-8,
    )


@pytest.mark.parametrize("mode", MODES)
def test_sizes_match_r(case: Case, mode: str) -> None:
    # To 1e-3 relative: R fits each size with `optimize`, whose default
    # tolerance is about 1e-4.
    got = _retro(case, **_aggregators(case, mode))
    np.testing.assert_allclose(
        got, read_vec(case.name, f"size_{mode}"), rtol=1e-3
    )


def test_fit_by_horizon_matches_per_column_fit() -> None:
    obs = np.round(np.abs(np.random.default_rng(0).normal(30, 6, (8, 3))))
    pred = np.full((8, 3), 30.0)
    got = fit_by_horizon(obs, pred)
    expected = [float(fit_nb(obs[:, i], pred[:, i])) for i in range(3)]
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("i", range(1, len(NB_FITS) + 1))
def test_fit_nb_matches_r_optimize(i: int) -> None:
    # To 1e-3 relative, as the sizes above.
    x, mu = NB_FITS[i - 1]
    expected = read("nb", f"fit_out_{i}").item()
    got = float(fit_nb(x, mu))
    np.testing.assert_allclose(got, expected, rtol=1e-3)


def test_fit_nb_empty_is_nan() -> None:
    assert np.isnan(float(fit_nb(np.array([]), np.array([]))))


def test_fit_nb_aborts() -> None:
    with pytest.raises(ValueError):
        fit_nb(np.array([-1.0, 2.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        fit_nb(np.array([1.0, 2.0]), np.array([-1.0, 1.0]))
    with pytest.raises(ValueError):
        fit_nb(np.array([1.5, 2.0]), np.array([1.0, 1.0]))


def test_retro_aborts(small: Case) -> None:
    pre = preprocess_negative_values(small.values, small.mask)
    with pytest.raises(ValueError):
        estimate_uncertainty_retro(pre, small.mask, small.n_hist, 1)
    with pytest.raises(ValueError):
        estimate_uncertainty_retro(
            pre, small.mask, small.mask.shape[0], small.mask.shape[0]
        )
    with pytest.raises(ValueError):
        estimate_uncertainty_retro(pre, small.mask, 1, 3)


def test_estimate_uncertainty_skips_failed_nowcasts() -> None:
    v, m = from_na(TOY)
    pre = preprocess_negative_values(v, m)
    from nowcastbaselines.baselinenowcast import (
        apply_reporting_structure,
        estimate_and_apply_delay,
        get_reporting_structure,
        truncate_to_rows,
    )

    trunc = truncate_to_rows(pre, m, n=3)
    s = get_reporting_structure(m)
    retro = []
    for tv, tm in trunc:
        rm = apply_reporting_structure(tv, tm, s)
        retro.append((jnp.where(jnp.asarray(rm), tv, 0.0), rm))
    ncs = [
        estimate_and_apply_delay(rv, rm, n=5, validate=False)
        for rv, rm in retro
    ]
    full = estimate_uncertainty(ncs, trunc, retro, n=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skipped = estimate_uncertainty(
            [ncs[0], None, ncs[2]], trunc, retro, n=3
        )
    assert full.shape == skipped.shape
    with pytest.raises(ValueError):
        estimate_uncertainty([None, None, None], trunc, retro, n=3)
    with pytest.raises(ValueError):
        estimate_uncertainty(ncs[:2], trunc, retro, n=3)


def _hub_week_inputs(
    cutoff: str, step: int, n_ref: int = 262, d: int = 84
) -> Any:
    import pandas as pd

    from nowcastbaselines.baselinenowcast import (
        estimate_and_apply_delay,
        hub_weeks,
    )

    rng = np.random.default_rng(1)
    pmf = np.exp(-np.arange(d + 1) / 10)
    pmf /= pmf.sum()
    full = rng.poisson(30 * pmf[None, :] * np.ones((n_ref, 1))).astype(float)

    def tri(c: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(c + 1)[:, None]
        dl = np.arange(d + 1)[None, :]
        m = (t + dl) <= c
        return np.where(m, full[: c + 1], 0.0), m

    first = pd.Timestamp(cutoff) - pd.Timedelta(days=n_ref - 1)
    lv, lm = tri(n_ref - 1)
    qs = [n_ref - 1 - step, n_ref - 1 - 2 * step]
    ncs, trs, rts = [], [], []
    for q in qs:
        v, m = tri(q)
        ncs.append(estimate_and_apply_delay(v, m, n=150))
        trs.append((lv[: q + 1], lm[: q + 1]))
        rts.append((v, m))
    return ncs, trs, rts, hub_weeks(first.date())


@pytest.mark.parametrize(("step", "horizons"), [(7, 12), (8, 13)])
def test_estimate_uncertainty_runs_with_hub_weeks(
    step: int, horizons: int
) -> None:
    # The cut-off is a Saturday, so the first retrospective origin is a
    # Saturday only when it is a whole number of weeks earlier.
    ncs, trs, rts, agg = _hub_week_inputs("2026-10-24", step)
    store, model = _capture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = estimate_uncertainty(
            ncs, trs, rts, n=2, uncertainty_model=model, ref_time_aggregator=agg
        )
    assert store["obs"].shape == (2, horizons)
    assert out.shape[0] == horizons
