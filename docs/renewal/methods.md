# Methods

How the renewal models used for the hub's nowcasts and forecasts work, from data to submission.
{ .page-subtitle }

`renewal` fits the health zones with recent cases jointly, pooled within provinces and linked by importation, with a renewal equation for infections and a reporting delay learnt jointly with the counts.
`renewalbvd` adds the reproduction number estimates of [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) as a down-weighted term, and `renewalsp` fits the same model to each zone on its own.
Their methods, equations, priors, inference and limitations are in the paper's [supporting information](../paper/si.html#renewal-model).

The code is in [`zones.py`](https://github.com/seabbs/bdbv-hub-modelling/blob/main/packages/nowcastrenewal/src/nowcastrenewal/zones.py) and [`model/`](https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/nowcastrenewal/src/nowcastrenewal/model).
How the model is built in code and sampled is on the [Implementation](../methods/implementation.md) page.
What each fit gives beyond the SI, such as the hub's weekly Rt target, is on the [Methods reference](../methods/model.md) page.

## Submission

The hub's target is `wk inc confirmed cases` as 100 samples for 4 weeks before the reference date to 3 weeks after it, for every zone, province and the country ([Hub specification](../submission/hub-spec.md)).
Zone trajectories go into the hub file as they are, and provinces and the country are their sums.
The submission walks a chain of candidates, `renewalbvd`, `renewalsp`, `renewal`, `randomwalknowcast` and `randomwalk`, and submits the first that passes its checks ([Submission workflow](../submission/workflow.md#renewal-chain)).
The sampler settings of each candidate are in `packages/bdbvsubmission/policy.toml`.
Each fit writes its convergence diagnostics to its `diagnostics.json`, which the fit store keeps.

## Development history

- Fits by province, before the zone fits, saturated the maximum tree depth in most draws, which made their convergence slow and fragile.
  The zone fits do not.
- A zone fit takes 1.5 to 2 times as long as a fit by province.
- In the backtest of the zone fits against the province fits, scored against each later release, the zone fits beat the province fits at every level.
  On weeks complete at the newest release they were mixed at the province level.
  Both did worse than `randomwalknowcast` at the national level at the two latest origins.
