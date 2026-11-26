# Baseline models

The random walk nowcast and the random walk.
{ .page-subtitle }

The random walk nowcast (`randomwalknowcast`) nowcasts the most recent weeks from the reporting delay of past releases, using our port of the baselinenowcast method, and forecasts each nowcast draw forward with a random walk on the log scale.
The random walk (`randomwalk`) is the same walk started from the latest complete week, with no nowcast.
Their methods, equations, differences between case studies and limitations are in the paper's [supporting information](../paper/si.html#baselines).

The code is in [nowcastbaselines](https://samabbott.co.uk/bdbv-hub-modelling/nowcastbaselines/), and which module does what is on the [Implementation](implementation.md#baseline-models) page.
How their province draws are paired is under [Correlated trajectories](evaluation.md#correlated-trajectories).
The data and notation are on the [Data methods page](data.md#target-and-notation).
