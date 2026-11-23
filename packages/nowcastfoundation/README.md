# nowcastfoundation

Chronos-2 nowcasts and forecasts from reporting triangles, and the code that fine-tunes Chronos-2 on simulated outbreaks.
It is for anyone using our fine-tuned checkpoints on their own data, and for working on the simulation, training and scoring steps of the workflow.

## Start here

- To nowcast your own triangle with a released checkpoint, read [`checkpoints`](src/nowcastfoundation/checkpoints.py) and the [Using the fine-tuned model](https://samabbott.co.uk/bdbv-hub-modelling/methods/finetuned-chronos/) page.
- To see how a forecast is made, read [`forecaster`](src/nowcastfoundation/forecaster.py) (`FoundationModel`).
- To simulate one batch: `uv run python -m nowcastfoundation.simulator --help`.

## Modules

- [`forecaster`](src/nowcastfoundation/forecaster.py): Chronos-2 forecasters on the hub target's releases, zero-shot or fine-tuned.
- [`backends`](src/nowcastfoundation/backends/): the Chronos-2 backend, with its device, precision and batched quantiles.
- [`checkpoints`](src/nowcastfoundation/checkpoints.py): loading a released checkpoint for your own reporting triangle.
- [`simulator`](src/nowcastfoundation/simulator/): the grid simulator. It draws each simulation's design and values (`draws`) and runs batches (`batch`) through the joint model in `nowcastrenewal.simulation`.
- [`finetune`](src/nowcastfoundation/finetune/): training blocks (`chronos`), the streaming trainer (`stream`), held-out scoring (`evaluate`) and hash checks (`verify`).
- [`series_bias`](src/nowcastfoundation/series_bias.py): the attention bias that ties a place's series together and to its parent.
- [`sidecar`](src/nowcastfoundation/sidecar.py): an offloaded checkpoint's held-out losses.
- [`checkpoint`](src/nowcastfoundation/checkpoint.py): scoring each published checkpoint against the references.
- [`live_checkpoint`](src/nowcastfoundation/live_checkpoint.py), [`live_checkpoint_page`](src/nowcastfoundation/live_checkpoint_page.py): the live run's newest checkpoints as preliminary results.
- [`simulate`](src/nowcastfoundation/simulate.py): province outbreaks published as hub-style releases, for backtests and the docs' examples.
- [`r/`](r/): an R package of the R simulation code. Tests load it to check the latent components against R.

## Next

- The workflow steps that run this package: [simulate](../../workflow/README.md#1-simulate-rulessimulatesmk), [fine-tune](../../workflow/README.md#2-fine-tune-rulesfinetunesmk), [publish](../../workflow/README.md#3-publish-rulespublishsmk) and [score](../../workflow/README.md#4-score-rulesscoresmk).
- The methods are in the [paper](https://samabbott.co.uk/bdbv-hub-modelling/paper/).
- The hub models built on it: [Foundation models](https://samabbott.co.uk/bdbv-hub-modelling/methods/foundation/).

```sh
uv add nowcastfoundation
```
