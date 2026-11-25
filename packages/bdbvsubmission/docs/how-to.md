# How to

The examples continue from [Getting started](index.md).

## Read the fallback chain

`policy.toml` lists the candidates in the order the gate tries them, and the thresholds of each check.
We write a short policy with one renewal model and the random walk.

```python
from bdbvsubmission.policy import fit_timeout, load_policy

policy_file = path.parent / "policy.toml"
policy_file.write_text(
    """
[timing]
fit_cap_minutes = 350
reserve_minutes = 45
min_fit_minutes = 30

[[candidates]]
id = "main"
kind = "main"
config = "main"
model = "renewal"
stage = "primary"
label = "zone renewal model"

[[candidates]]
id = "randomwalk"
kind = "randomwalk"
config = "randomwalk"
stage = "baseline"
label = "random walk baseline"

[convergence.fail]
rhat = 1.1
ess_bulk = 25
ess_tail = 25
divergent_fraction = 0.05

[convergence.warn]
rhat = 1.05
ess_bulk = 100
ess_tail = 100
divergent_fraction = 0.01

[ppc]
stream = "total"

[checks]
samples = 100
"""
)
policy = load_policy(policy_file)
assert [c.id for c in policy.chain()] == ["main", "randomwalk"]
```

A fit's timeout is the job cap, less the time kept free for the gate before the window closes.

```python
assert fit_timeout(policy, None) == 350
assert fit_timeout(policy, minutes_left=120) == 75
assert fit_timeout(policy, minutes_left=50) == 30
```

## Gate a fit on its convergence

`verdict` compares a fit's headline diagnostics with the policy's thresholds.
Past a *fail* threshold the fit is ruled out and the chain moves on.
Past a *warn* threshold it is reported only.

```python
from bdbvsubmission.summarise_fit import verdict

head = {
    "max_rhat": 1.07,
    "min_ess_bulk": 400.0,
    "min_ess_tail": 300.0,
    "divergent_fraction": 0.0,
}
status, failures, warnings = verdict(head, policy.convergence)
assert status == "warn"
assert failures == []
assert len(warnings) == 1
```

## Judge a forecast against a count

The plausibility check compares a forecast with the recent reported level.
`poisson_limits` gives the exact interval of a Poisson mean from one count.

```python
from bdbvsubmission.check_forecast import poisson_limits

lower, upper = poisson_limits(10, 0.975)
assert 4.7 < lower < 4.9
assert 18.3 < upper < 18.5
```

Next: [Reference](reference.md).
