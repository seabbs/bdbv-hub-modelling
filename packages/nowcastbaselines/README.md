# nowcastbaselines

The random walk and the random walk nowcast, the baselines every other model is compared with.
It is for anyone who needs a simple, well-tested forecast of weekly counts revised as later releases report more cases.

## Start here

- Read [`randomwalk`](src/nowcastbaselines/randomwalk.py) (`RandomWalk`, `RandomWalkNowcast`).
- The [docs site](https://samabbott.co.uk/bdbv-hub-modelling/nowcastbaselines/) has a worked example, how-to recipes and the API reference.

## Modules

- [`randomwalk`](src/nowcastbaselines/randomwalk.py): the random walk forecaster and its nowcasting variant.
- [`chain`](src/nowcastbaselines/chain.py), [`walk`](src/nowcastbaselines/walk.py), [`obs`](src/nowcastbaselines/obs.py): the chain-ladder delay fit, random walks on the log scale, and the observation model.
- [`baselinenowcast`](src/nowcastbaselines/baselinenowcast/): a JAX port of the baselinenowcast R package.
- [`pooling`](src/nowcastbaselines/pooling.py), [`introductions`](src/nowcastbaselines/introductions.py), [`copula`](src/nowcastbaselines/copula.py), [`zones`](src/nowcastbaselines/zones.py): pooling over a hierarchy, introductions into silent places, a copula pairing places, and splitting places into zones.

## Next

- [How it is used in the workflow](https://samabbott.co.uk/bdbv-hub-modelling/nowcastbaselines/workflow/).
- The baselines in the hub submission: [Baseline models](https://samabbott.co.uk/bdbv-hub-modelling/methods/baselines/).

```sh
uv add nowcastbaselines
```
