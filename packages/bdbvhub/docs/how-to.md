# How to

The examples continue from [Getting started](index.md).

## Find the registered models

Importing `bdbvhub.models` registers every built-in model: the random walks, the Chronos-2 models and the renewal variants.
`get_model` builds one by name.

```python
from bdbvhub.models import list_models

names = {spec.name for spec in list_models()}
assert {"randomwalk", "randomwalknowcast", "renewal", "foundation"} <= names
```

## Look up a submission fit configuration

The submission CI fits each candidate in its fallback chain with `bdbvhub fit --config <name>`.
Each name maps to a `FitConfig` that says how to fit the model and how to forecast from the fit.

```python
from bdbvhub.submission.fitting import get_config

config = get_config("randomwalk")
assert config.name == "randomwalk"
print(config.description)
```

## Check a submission file name

The hub reads the round, team and model from each file name.
`parse_file_name` splits a name as the hub's own checks do.

```python
from bdbvhub.submission.validate import parse_file_name

parts = parse_file_name(path.name)
assert parts == {
    "round_id": "2026-03-14",
    "team": "seabbs",
    "model": "rwnowcast",
    "ext": "csv",
}
```

`validate_file` runs the full checks against a local clone of the hub (`Hub.load`), as `bdbvhub validate` does.

## Build model metadata

`render_model_metadata` builds a model's metadata record in the hub's field order, with our team's shared fields filled in.
`MODEL_METADATA` holds the text of each model that may be submitted.

```python
from bdbvhub.submission.hub import render_model_metadata
from bdbvhub.submission.metadata import MODEL_METADATA

record = render_model_metadata("randomwalk", **MODEL_METADATA["randomwalk"])
assert record["team_abbr"] == "seabbs"
assert record["model_abbr"] == "randomwalk"
```

## Label weeks a fallback went out under our hub id

When the first model in the chain fails its checks, a fallback goes out under the same hub id.
`split_substituted` gives those weeks their own label so the comparison with the hub can show them apart.

```python
from bdbvhub.compare import split_substituted

forecasts = pd.DataFrame(
    {
        "model": ["seabbs-renewal", "seabbs-renewal"],
        "reference_date": ["2026-03-07", "2026-03-14"],
    }
)
ledger = pd.DataFrame(
    {
        "reference_date": ["2026-03-14"],
        "hub_id": ["seabbs-renewal"],
        "model": ["renewalsp"],
        "substituted": ["true"],
    }
)
relabelled = split_substituted(forecasts, ledger)
assert relabelled["model"].tolist() == [
    "seabbs-renewal",
    "seabbs-renewal [renewalsp]",
]
```

Next: [Reference](reference.md).
