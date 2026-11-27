# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Command line interface: `nowcastgermany <command>`.

The German hospitalisation nowcasting benchmark
(packages/nowcastgermany/config/germany.toml):

- `fetch`: download the benchmark's data at the config's commits.
- `backtest`: nowcast and score the benchmark with our models.
- `teams`: score the hub teams' nowcasts.
- `asof`: write the 7-day sums as each development date saw them.
- `paper`: write the forecasts as the paper's long table.
- `units`: write our models' development forecasts and scores from the
  fine-tuning workflow's German units (`nowcastgermany.store`), the
  paper's development table.
- `stored-load`: write them from the published `paper-germany` release,
  for Site results.
- `pairing-sensitivity`: score the walk models with and without their
  pairing on the development half (`nowcastgermany.pairing`).

Run from the repo root. The data default to `$BDBV_DATA_DIR/raw/germany`
(`data/raw/germany`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from nowcastgermany import evaluation as de
from nowcastgermany import store as gs
from nowcastgermany.data import hub_triangle
from nowcastgermany.pairing import germany_sensitivity
from nowcastutils.chunks import CHUNK_HELP, chunk
from nowcastutils.files import write_csv


def _data_dir() -> str:
    """Return the German data directory under `$BDBV_DATA_DIR`."""
    return f"{os.environ.get('BDBV_DATA_DIR') or 'data'}/raw/germany"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(prog="nowcastgermany")
    sub = parser.add_subparsers(dest="command", required=True)
    data_dir = _data_dir()

    gfe = sub.add_parser(
        "fetch",
        help="Download the German hospitalisation benchmark's data.",
    )
    gfe.add_argument(
        "--config", default="packages/nowcastgermany/config/germany.toml"
    )
    gfe.add_argument("--out", default=data_dir)

    gbt = sub.add_parser(
        "backtest",
        help="Nowcast and score the German hospitalisation benchmark.",
    )
    gbt.add_argument(
        "--config", default="packages/nowcastgermany/config/germany.toml"
    )
    gbt.add_argument("--data", default=data_dir)
    gbt.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Defaults to the benchmark's models in the config, with its "
            "held_out_models on the held-out half."
        ),
    )
    gbt.add_argument(
        "--chunk", default=None, help=CHUNK_HELP.format(what="nowcast dates")
    )
    gbt.add_argument("--out", default="results/germany")
    gbt.add_argument(
        "--held-out",
        action="store_true",
        help="Run the held-out half of the dates, for the paper, into "
        "OUT/held-out.",
    )

    pst = sub.add_parser(
        "pairing-sensitivity",
        help="Score randomwalk and randomwalknowcast with no pairing and "
        "with their states paired by the rolling rho_p, from one forecast "
        "each, on the development half, with the rho_p of each task.",
    )
    pst.add_argument("--data", default=data_dir)
    pst.add_argument(
        "--config", default="packages/nowcastgermany/config/germany.toml"
    )
    pst.add_argument(
        "--chunk", default=None, help=CHUNK_HELP.format(what="nowcast dates")
    )
    pst.add_argument("--out", default="results/pairing")

    gtm = sub.add_parser(
        "teams",
        help="Score the German hospitalisation hub teams' nowcasts.",
    )
    gtm.add_argument(
        "--config", default="packages/nowcastgermany/config/germany.toml"
    )
    gtm.add_argument("--data", default=data_dir)
    gtm.add_argument("--out", default="results/germany")
    gtm.add_argument(
        "--held-out",
        action="store_true",
        help="Score the held-out half of the dates, for the paper, into "
        "OUT/held-out.",
    )

    gas = sub.add_parser(
        "asof",
        help="Write the German 7-day sums as each development date saw them.",
    )
    gas.add_argument(
        "--config", default="packages/nowcastgermany/config/germany.toml"
    )
    gas.add_argument("--data", default=data_dir)
    gas.add_argument("--out", default="results/germany")

    gpp = sub.add_parser(
        "paper",
        help="Write the German benchmark's forecasts as the paper's table.",
    )
    gpp.add_argument(
        "--config", default="packages/nowcastgermany/config/germany.toml"
    )
    gpp.add_argument("--results", default="results/germany")

    gun = sub.add_parser(
        "units",
        help="Write our models' German development forecasts and scores "
        "from the fine-tuning workflow's units (`nowcastgermany.store`).",
    )
    gun.add_argument(
        "units",
        nargs="+",
        help="The units' folders: the reference and the checkpoint's.",
    )
    gun.add_argument(
        "--config", default="packages/nowcastgermany/config/germany.toml"
    )
    gun.add_argument("--out", default="results/germany")
    gun.add_argument(
        "--record",
        default=None,
        help="The checkpoint's release record (published.json), which the "
        "note names.",
    )

    gsl = sub.add_parser(
        "stored-load",
        help="Write our models' German development forecasts and scores "
        "from the published paper-germany release, for Site results.",
    )
    gsl.add_argument(
        "--config", default="packages/nowcastgermany/config/germany.toml"
    )
    gsl.add_argument(
        "--store",
        required=True,
        help="The unpacked paper-germany release (`paperpipe.stored`).",
    )
    gsl.add_argument(
        "--out", required=True, help="A `backtest` results directory."
    )
    gsl.add_argument(
        "--note",
        required=True,
        help="JSON naming the release and checkpoint shown, which the page "
        "reads.",
    )

    return parser


def _backtest(args: argparse.Namespace) -> int:

    config = de.load_germany_config(args.config)
    dates = de.nowcast_dates(config, held_out=args.held_out)
    if args.chunk:
        dates = chunk(dates, args.chunk)
    models = (
        de.run_models(config, args.held_out)
        if args.models is None
        else args.models
    )
    triangle = hub_triangle(Path(args.data) / Path(config.triangle).name)
    timings: list[dict[str, Any]] = []
    forecasts = de.with_truth(
        de.germany_forecasts(
            models,
            triangle,
            dates,
            config.n_samples,
            config.seed,
            max_delay=config.truth_delay,
            timings=timings,
        ),
        triangle,
        config.truth_delay,
    )
    scores = de.score_germany(forecasts, config.truth_delay)
    out = Path(args.out) / ("held-out" if args.held_out else "")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(forecasts, out / "forecasts.csv.gz")
    write_csv(scores, out / "scores.csv.gz")
    # Seconds by model and date (`germany_forecasts`), for the
    # `benchmarks` workflow target.
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
    print(f"German benchmark: {len(scores)} scored nowcasts in {out}.")
    return 0


def _pairing_sensitivity(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "pairing-germany"
    config = args.config
    data = args.data
    dates = de.nowcast_dates(de.load_germany_config(config))
    if args.chunk:
        dates = chunk(dates, args.chunk)
    # Each date's scores and rho_p are kept as it is scored, so a rerun
    # resumes.
    done = out / "germany-dates"
    done.mkdir(exist_ok=True)
    for date in dates:
        path, rho_path = done / f"{date}.csv.gz", done / f"{date}-rho.csv.gz"
        if not (path.exists() and rho_path.exists()):
            scored, rho = germany_sensitivity(data, config, [date])
            write_csv(scored, path)
            write_csv(rho, rho_path)
    scores = pd.concat(
        [_read_or_empty(done / f"{d}.csv.gz") for d in dates],
        ignore_index=True,
    )
    rho = pd.concat(
        [_read_or_empty(done / f"{d}-rho.csv.gz") for d in dates],
        ignore_index=True,
    )
    if args.chunk:
        name += "-" + args.chunk.replace("/", "of")
    path = out / f"{name}.csv.gz"
    write_csv(scores, path)
    write_csv(rho, out / f"{name}-rho.csv.gz")
    print(f"Pairing sensitivity (germany): {len(scores)} rows in {path}.")
    return 0


def _read_or_empty(path: Path) -> pd.DataFrame:
    """Read a results file, or an empty table if it has no columns."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _teams(args: argparse.Namespace) -> int:

    config = de.load_germany_config(args.config)
    data = Path(args.data)
    submissions = de.read_submissions(
        [data / Path(p).name for p in config.submissions]
    )
    dates = de.nowcast_dates(config, held_out=args.held_out)
    forecasts = de.with_truth(
        de.team_forecasts(submissions, dates),
        hub_triangle(data / Path(config.triangle).name),
        config.truth_delay,
    )
    scores = de.score_germany(forecasts, config.truth_delay)
    out = Path(args.out) / ("held-out" if args.held_out else "")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(forecasts, out / "teams-forecasts.csv.gz")
    write_csv(scores, out / "teams.csv.gz")
    print(f"German hub teams: {len(scores)} scored nowcasts in {out}.")
    return 0


def _asof(args: argparse.Namespace) -> int:

    config = de.load_germany_config(args.config)
    triangle = hub_triangle(Path(args.data) / Path(config.triangle).name)
    asof = de.asof_sums(triangle, de.nowcast_dates(config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(asof, out / "asof.csv.gz")
    print(f"German data as of each date: {len(asof)} sums in {out}.")
    return 0


def _paper(args: argparse.Namespace) -> int:

    config = de.load_germany_config(args.config)
    res = Path(args.results)
    parts = [
        pd.read_csv(path, dtype={"origin": str, "target_end_date": str})
        for name in ("forecasts.csv.gz", "teams-forecasts.csv.gz")
        if (path := res / name).exists()
    ]
    paper = de.paper_forecasts(pd.concat(parts, ignore_index=True))
    write_csv(paper, res / "paper-forecasts.csv.gz")
    meta = {
        "study": "germany",
        "split_date": config.split_date,
        "baseline": config.paper_baseline,
        "truth_delay": config.truth_delay,
        "quantile_levels": list(de.QUANTILE_LEVELS),
    }
    (res / "paper-forecasts.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Paper table: {len(paper)} rows in {res}.")
    return 0


def _write_ours(
    forecasts: pd.DataFrame, config: de.GermanyConfig, out: Path
) -> None:
    """Write our forecasts and their scores, as `backtest` does."""
    scores = de.score_germany(forecasts, config.truth_delay)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(forecasts, out / "forecasts.csv.gz")
    write_csv(scores, out / "scores.csv.gz")
    print(f"Our German forecasts: {len(scores)} scored nowcasts in {out}.")


def _units(args: argparse.Namespace) -> int:
    config = de.load_germany_config(args.config)
    forecasts = gs.units_forecasts(args.units, config.split_date, config.models)
    out = Path(args.out)
    _write_ours(forecasts, config, out)
    note = {
        "checkpoint": gs.checkpoint_of(args.record),
        "models": sorted(set(forecasts["model"])),
    }
    (out / gs.NOTE).write_text(json.dumps(note, indent=2) + "\n")
    return 0


def _stored_load(args: argparse.Namespace) -> int:
    config = de.load_germany_config(args.config)
    store = Path(args.store)
    note = Path(args.note)
    note.parent.mkdir(parents=True, exist_ok=True)
    if not (store / gs.UNITS).exists():
        note.write_text(json.dumps({"release": None}) + "\n")
        print(f"No published German forecasts of ours in {store}.")
        return 0
    manifest = json.loads((store / "manifest.json").read_text())
    made = json.loads((store / gs.UNITS_NOTE).read_text())
    forecasts = gs.load_units(store, config.split_date, config.models)
    _write_ours(forecasts, config, Path(args.out))
    meta = {"release": manifest["tag"], **made}
    note.write_text(json.dumps(meta, indent=2) + "\n")
    return 0


COMMANDS = {
    "backtest": _backtest,
    "teams": _teams,
    "asof": _asof,
    "paper": _paper,
    "units": _units,
    "stored-load": _stored_load,
    "pairing-sensitivity": _pairing_sensitivity,
}
"""Each command's handler, but `fetch` (network)."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments. Defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        The exit code.
    """
    args = build_parser().parse_args(argv)
    if args.command == "fetch":  # pragma: no cover - network
        de.fetch_germany(de.load_germany_config(args.config), args.out)
        print(f"German benchmark data written to {args.out}.")
        return 0
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
