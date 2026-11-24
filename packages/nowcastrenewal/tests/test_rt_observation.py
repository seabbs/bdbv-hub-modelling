import math

import jax.numpy as jnp
import numpyro
import pytest

from nowcastrenewal.model.rt_observation import RtPseudoObservation

ARGS = {
    "day": 7,
    "groups": [[True, True], [True, False]],
    "values": [0.0, 0.1],
    "scales": [0.1, 0.2],
    "kinds": [0, 1],
}


def sites(sd_inflation):
    obs = RtPseudoObservation("bvd", sd_inflation, **ARGS)
    rt = jnp.ones((21, 2))
    infections = jnp.ones((21, 2))
    with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=0):
        obs(rt=rt, infections=infections, n_init=0)
    return tr


def test_sites_are_named_from_the_name():
    tr = sites(1.5)
    assert set(tr) == {"bvd_log_rt_model", "bvd_z", "bvd_rt_obs"}
    assert tr["bvd_rt_obs"]["is_observed"]


def test_an_infinite_sd_inflation_adds_no_likelihood():
    assert "bvd_rt_obs" not in sites(math.inf)


def test_the_sd_inflation_multiplies_the_external_sds():
    tr = sites(1.5)
    scale = tr["bvd_rt_obs"]["fn"].scale
    assert jnp.allclose(scale, 1.5 * jnp.asarray(ARGS["scales"]))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sd_inflation": 0.9}, "sd_inflation"),
        ({"sd_inflation": math.nan}, "sd_inflation"),
        ({"day": -1}, "day"),
        ({"groups": [True, False]}, "groups"),
        ({"values": [0.0]}, "values"),
        ({"scales": [0.1, 0.0]}, "positive"),
        ({"kinds": [0, 2]}, "kinds"),
    ],
)
def test_bad_inputs_raise(change, message):
    args = {"sd_inflation": 1.5, **ARGS, **change}
    with pytest.raises(ValueError, match=message):
        RtPseudoObservation("bvd", **args)


def test_repr_names_the_component():
    obs = RtPseudoObservation("bvd", 1.5, **ARGS)
    assert repr(obs) == (
        "RtPseudoObservation(name='bvd', sd_inflation=1.5, day=7, n_groups=2)"
    )


def test_group_rt_weights_each_place_by_its_infections():
    from nowcastrenewal.model.rt_observation import group_rt

    rt = jnp.array([1.0, 2.0, 4.0])
    weight = jnp.array([1.0, 3.0, 0.0])
    members = jnp.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    out = group_rt(rt, weight, members)
    assert float(out[0]) == pytest.approx(7.0 / 4.0)
    assert math.isnan(float(out[1]))
    assert float(out[2]) == pytest.approx(7.0 / 4.0)
