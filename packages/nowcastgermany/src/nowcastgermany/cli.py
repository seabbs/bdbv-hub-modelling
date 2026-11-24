# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Command line interface: `nowcastgermany <command>`.

The German hospitalisation nowcasting benchmark (config/germany.toml):

- `fetch`: download the benchmark's data at the config's commits.
- `backtest`: nowcast and score the benchmark with our models.
- `teams`: score the hub teams' nowcasts.
- `asof`: write the 7-day sums as each development date saw them.
- `paper`: write the forecasts as the paper's long table.
- `ft-load`: write the stored forecasts of the pinned fine-tuned
  checkpoint, for the Pages build.
- `ft-tag`: print the pinned checkpoint's release tag, whose asset the
  Pages build downloads.
- `unit-publish`: attach a released checkpoint's German unit to its
  release as the stored forecasts (`publish_germany_unit`).
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

from nowcastfoundation import forecaster as foundation
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
    gfe.add_argument("--config", default="config/germany.toml")
    gfe.add_argument("--out", default=data_dir)

    gbt = sub.add_parser(
        "backtest",
        help="Nowcast and score the German hospitalisation benchmark.",
    )
    gbt.add_argument("--config", default="config/germany.toml")
    gbt.add_argument("--data", default=f"{data_dir}/raw/germany")
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
    pst.add_argument("--config", default="config/germany.toml")
    pst.add_argument(
        "--chunk", default=None, help=CHUNK_HELP.format(what="nowcast dates")
    )
    pst.add_argument("--out", default="results/pairing")

    gtm = sub.add_parser(
        "teams",
        help="Score the German hospitalisation hub teams' nowcasts.",
    )
    gtm.add_argument("--config", default="config/germany.toml")
    gtm.add_argument("--data", default=f"{data_dir}/raw/germany")
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
    gas.add_argument("--config", default="config/germany.toml")
    gas.add_argument("--data", default=f"{data_dir}/raw/germany")
    gas.add_argument("--out", default="results/germany")

    gpp = sub.add_parser(
        "paper",
        help="Write the German benchmark's forecasts as the paper's table.",
    )
    gpp.add_argument("--config", default="config/germany.toml")
    gpp.add_argument("--results", default="results/germany")

    gft = sub.add_parser(
        "ft-load",
        help="Write the stored German forecasts of the fine-tuned model's "
        "checkpoint, for the Pages build (`nowcastgermany.store`).",
    )
    gft.add_argument("--config", default="config/germany.toml")
    gft.add_argument("--finetune", default="config/finetune.toml")
    gft.add_argument(
        "--store",
        required=True,
        help="The unpacked release asset (`nowcastgermany.store.ASSET`).",
    )
    gft.add_argument(
        "--out", required=True, help="A `backtest` results directory."
    )
    gft.add_argument(
        "--note",
        required=True,
        help="JSON naming the checkpoint shown, which the page reads.",
    )

    gtg = sub.add_parser(
        "ft-tag",
        help="Print the release tag of the checkpoint config/finetune.toml "
        "pins.",
    )
    gtg.add_argument("--finetune", default="config/finetune.toml")

    gup = sub.add_parser(
        "unit-publish",
        help="Attach a released checkpoint's German unit to its release as "
        "the stored forecasts (`nowcastgermany.store.pack_unit`).",
    )
    gup.add_argument("cache", help="The score cache.")
    gup.add_argument("label", help="The checkpoint's label, slots-<hash>.")
    gup.add_argument("--config", default="config/germany.toml")
    gup.add_argument("--repo", default="seabbs/bdbv-hub-modelling")
    gup.add_argument(
        "--out", default=None, help="Write the asset here and upload nothing."
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


def _ft_load(args: argparse.Namespace) -> int:
    config = de.load_germany_config(args.config)
    tag = gs.pinned_tag(foundation.finetuned_spec(args.finetune))
    stored = gs.stored_tags(args.store)
    note = Path(args.note)
    note.parent.mkdir(parents=True, exist_ok=True)
    shown = tag if tag in stored else None
    meta = {"tag": shown, "pinned": tag, "models": list(gs.STORED_MODELS)}
    note.write_text(json.dumps(meta, indent=2) + "\n")
    if shown is None:
        print(f"No stored German forecasts of the _ft models for {tag}.")
        return 0
    forecasts = gs.load_stored(args.store, tag, config.split_date)
    scores = de.score_germany(forecasts, config.truth_delay)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(forecasts, out / "forecasts.csv.gz")
    write_csv(scores, out / "scores.csv.gz")
    print(f"The _ft models at {tag}: {len(scores)} scored nowcasts in {out}.")
    return 0


def _ft_tag(args: argparse.Namespace) -> int:
    print(gs.pinned_tag(foundation.finetuned_spec(args.finetune)))
    return 0


def _unit_publish(args: argparse.Namespace) -> int:  # pragma: no cover - gh
    import subprocess
    import tempfile

    config = de.load_germany_config(args.config)
    tag = gs.release_tag(args.label)
    if args.out:
        path = gs.pack_unit(args.cache, args.label, args.out, config.split_date)
        print(f"Wrote {path}.")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        path = gs.pack_unit(args.cache, args.label, tmp, config.split_date)
        subprocess.run(
            ["gh", "release", "upload", tag, str(path), "--clobber",
             "-R", args.repo],
            check=True,
        )  # fmt: skip
    print(f"Attached {gs.ASSET} to {tag}.")
    return 0


COMMANDS = {
    "backtest": _backtest,
    "teams": _teams,
    "asof": _asof,
    "paper": _paper,
    "ft-load": _ft_load,
    "ft-tag": _ft_tag,
    "unit-publish": _unit_publish,
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
