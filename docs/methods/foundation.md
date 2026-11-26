# Foundation models

The `foundation` model and its comparison variants give the hub target's releases to the Chronos-2 time series foundation model[^chronos2].
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
`foundation` is Chronos-2 fine-tuned on simulated outbreaks ([Fine-tuning](#fine-tuning)) on the reporting-triangle layout.
The variants the paper compares it with are `foundation_zeroshot` (pretrained weights), `foundation_zeroshot_complete` and `foundation_complete` (complete layout), `foundation_baselinenowcast` and `foundation_zeroshot_baselinenowcast` (baselinenowcast), and `foundation_real` (fine-tuned on real data alone).
None is fitted to these data: Chronos-2 forecasts in context, from its pretrained weights or a fine-tuned checkpoint.
Only `foundation` runs in the Fit store and the submission; the variants are run by the paper's workflow alone.
Each model reads the cadence from the data: one pass on the daily releases gives the daily counts and the weekly hub target, and a task with weekly releases only gives the weekly target.
The hub model `seabbs-foundation` is set up to submit `foundation` and is off until it is opted in ([Opting in](../submission/opting-in.md)).

In brief, each model:

1. lays out the hub target of every release by place, step and release lag ([Releases](#releases));
2. puts every hub target, the country, the provinces and their health zones, each as its own series, in one block that Chronos-2 forecasts jointly ([Layouts](#layouts));
3. draws 100 trajectories from Chronos-2's quantiles ([Trajectories](#trajectories));
4. rounds the draws to counts ([Counts](#counts)).

## Releases

The target is the hub's: confirmed cases by onset week, known and imputed onsets together.
Two sources give it with its updates; a model takes the daily one when the task has daily releases.

- **Daily**: the daily counts of each release, known plus imputed onsets.
  A step is an onset day.
- **Weekly**: the weekly counts of each release of the target data, for the country, the provinces and the zones.
  A step is an onset week.

For place $p$, step $t$ and release $r$ with cut-off in step $c_r$, the release lag is $d = c_r - t$.
$Y_{p,t,d}$ is the count of step $t$ at lag $d$, from the release whose cut-off is $d$ steps after $t$.
Release $r$ gives lag $d$ only at step $c_r - d$, so each lag has at most one value per release and is missing at other steps.
$X_{p,t}$ is the count of step $t$ at the vintage, with cut-off step $c$.

## Layouts

Every hub target is a series in one block: the country, all 26 provinces, and every health zone of those provinces.
Each is the hub target itself, so the country is not the sum of the provinces and the zones are not split from them.
Zone counts are zero before the zone releases start.
Chronos-2 forecasts every target series in the block jointly, from one shared last step of context, $t^*$, which each layout sets differently.

- **Complete** (`*_complete`): the targets $X_{p,t}$ for $t \le t^*$.
  $t^*$ is the earliest over places of the week `randomwalk` starts its walk from, the latest week 95% complete under its delay ([supporting information](../paper/si.html#baselines)).
  It comes from the delay alone, so it needs no walk spread.
- **Reporting triangle** (`*_ragged`): the same targets, with one lag series per place with cases and lag $d < D$, indexed by onset step as in a regression nowcast.
  Lag series $d$ holds $Y_{p,t,d}$, missing where it is not yet reported, and Chronos-2 masks the missing values.
  $t^* = c - D$, the last step whose count is at lag $D$ or more, with $D$ estimated from the data (`lag_count`).
  $D$ is the first lag at which the steps 28 to 112 days (4 to 16 weeks) before the cut-off had 99% of their counts reported, pooled over places and releases, read from no release after the cut-off.
  It is at most 112 days or 16 weeks and at most what the history allows, with two steps of context before the longest lag, and that many when the window has no counts or never reaches 99%.
  A model can fix $D$ instead (`lags`); the SI compares the estimate with 4 weeks or 28 days.
  The lag series run past $t^*$ as known future values, which is how recent counts reach the nowcast.
  A place with no cases has no lag series, as they would be all zero.
  The daily block also has each place's 7-day rolling sum as a series (`rolling7`), so the hub's 7-day target is a series of the block.
  So the daily block, where 75 of 546 places have cases, fits in memory.

Chronos-2 standardises each series and takes its arcsinh, so counts go in as they are.

## Trajectories

Chronos-2 gives 21 quantiles, from 1% to 99%, of each series at each step ahead.
Each of the 100 trajectories reads each series and step's quantile function at a uniform level: linear between the quantiles and flat beyond them.
The levels are coupled at each forecast by one step, `nowcastutils.trajectories.choose_coupling`, which chooses the coupling from the data.
Its candidates are a Gaussian copula (`couple`), with correlation `rho` between consecutive steps of a series and `s` between series, fitted (`fit_coupling`) over the earlier releases of the latest 14, 28 or 56 days, the Schaake shuffle, each sample ordered as one past window of the series' own reported values (`history_windows`), and comonotone draws (one level per sample and series).
The earlier releases are every release in the 70 days before the task's (`coupling_releases`), and the model forecasts the same block at each.
Each of those forecasts is scored at the steps that are final by the task's release, as normal scores of its randomised PIT (`pit_scores`), which the Gaussian fit reads.
Each release with at least two before it is a test: every candidate is fitted from what came before it, its draws are scored against what has been reported since (energy score), and the candidate with the lowest mean score is fitted from everything before the task and used.
A candidate that cannot be fitted at one of the tests (the Schaake shuffle with too little history) drops out.
With nothing to compare on, the Gaussian over the longest window is used when there are two releases to fit it from, else the Schaake shuffle, else independence.
On the German development half the step's draws give sums over the nowcast days 9 to 15% lower CRPS than independent draws (`nowcasteval.coupling`).
The chosen coupling, its fit and each candidate's score are written beside a submission (`copula.json`).
The candidate windows are the only settings; which is used is chosen at each forecast.
Independent and comonotone draws remain as named options (`coupling`) for the supplement.

## Counts

A draw is rounded to the nearest count and floored at zero.
Steps up to $t^*$ are their count at the vintage.
The hub's target is weekly.
A daily reporting-triangle model reads each week from its place's rolling 7-day series on the week's last day, and a daily complete model sums its seven daily draws.

## Patches and context

Chronos-2 reads a series in patches of 16 steps.
The context is every step from the first onset in the releases to $t^*$.

## Scoring

The models forecast final counts and model no delay.
So the release backtest scores them on the complete target only, each week against the first release the truth delay after it ends ([Evaluation](evaluation.md#real-releases)).

## Health zones

The models forecast the zones themselves, as series of the block.
The country, provinces and zones all come straight from the block, daily or weekly as the model runs.
Nothing is summed or split, so zones need not sum to their province.
Each zone keeps its published count, as every series does.
So the stored forecasts, the hub file and the scores all have the zones as submitted.
The renewal models fit the zones and sum them to their provinces instead ([Health zones](model.md#health-zones)).

## Fine-tuning

The fine-tuned models (`foundation`, `foundation_complete`, `foundation_baselinenowcast`) load one LoRA checkpoint of Chronos-2, fine-tuned on simulated outbreaks whose reporting follows epinowcast's observation model.
The simulations, the training and the checkpoints' scores are in the paper's [supporting information](../paper/si.html#fine-tuning), and how to load a checkpoint is on [Using the fine-tuned model](finetuned-chronos.md).
The simulator is `nowcastfoundation.simulator` (the design and value draws, then each simulation as the joint PyRenew model in `nowcastrenewal.simulation.outbreak`), run on the grid config `packages/nowcastfoundation/config/finetune/grid.yaml` by the workflow's `simulate_batch` rule; the training blocks are cut by `nowcastfoundation.finetune.training_block`, and the block design and release cadences are `BLOCK_DESIGN` in `nowcastfoundation.finetune.chronos` and `finetune.release_days`; the [workflow README](https://github.com/seabbs/bdbv-hub-modelling/blob/main/workflow/README.md) says which rounds a run trains on.

A checkpoint is named by its settings hash, of the blocks, the code and the training settings.
The first release, `chronos-ft-2c2b22a03107`, was trained from earlier code, so its hash cannot be reproduced from this repository.
Later releases are trained from the code here, so retraining from it gives the same hash.
Each checkpoint is published as the repository's release `chronos-slots-<hash>` (with the series bias) or `chronos-ft-<hash>`, with a `sha256sums.txt` of its files.
The models load the newest such release by publish date (`hash = "latest"` in `packages/nowcastfoundation/config/finetune.toml`), so each new release, interim or final, is picked up; a settings hash there pins one release instead.
`source` in `packages/nowcastfoundation/config/finetune.toml` says where the checkpoint is read from: `github` (the release), `huggingface` (a Hugging Face Hub repository at a pinned revision holding the same files) or `local` (a training segment folder).
A checkpoint read from the Hub is the same checkpoint as its release, so it keys the same stored fits.
The checkpoint's hash and adapter SHA-256 key the stored fits, so a new release refits the fine-tuned models, and each stored fit records the checkpoint it used.
The adapter is downloaded once and checked against `sha256sums.txt` and the settings hash.
The fit store resolves the release once per run, with the workflow's GitHub token, and passes its hash to every fit in `BDBV_CHRONOS_FT`.
If GitHub cannot be reached, building a model fails rather than fall back to a checkpoint downloaded before.
A fit whose adapter cannot be downloaded is stored as failed, and the fit store keeps it until its inputs change.
Such fits are retried by running the Fit store workflow with `retry_failed` (`gh workflow run fit-store.yml -R seabbs/bdbv-hub-modelling -f retry_failed=true -f models="foundation foundation"`).

## Limitations

- The zero-shot models use Chronos-2 as pretrained. It has not seen these data or outbreaks like them.
- The fine-tuned models have seen only simulated outbreaks, whose delays and dynamics may not match these data.
- The fine-tuned models were trained on blocks no larger than the memory budget allows, but forecast all 546 hub targets in one block.
- Draws are independent across steps and places, so a weekly value of the daily complete model, a sum of seven daily draws, is too narrow, as are sums over weeks or places.

[^chronos2]: Ansari A F et al. (2025) Chronos-2: From univariate to universal forecasting. arXiv:2510.15821. Code: <https://github.com/amazon-science/chronos-forecasting>.
