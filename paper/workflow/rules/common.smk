# The paper workflow's settings, kept apart from its rules
# (paper/workflow/Snakefile). Paths and arguments come from
# `paperpipe.flow.PaperFlow`.

# This file is a module of the project workflow (workflow/Snakefile), which
# runs from the repository root; paper/ is two folders up from this file.
PAPER = Path(workflow.current_basedir).parents[1]


sys.path.insert(0, str(PAPER))


sys.path.append(str(PAPER.parent / "workflow"))


from flowlib import forecast, localconfig  # noqa: E402


from paperpipe.config import MODES  # noqa: E402


from paperpipe.evidence import EVIDENCE as EVIDENCE_NAMES  # noqa: E402


from paperpipe import stored  # noqa: E402


from paperpipe.flow import BDBV, GERMANY_CLI, PYTHON, PaperFlow  # noqa: E402

# Evidence rows (`evidence` in the config): a file the workflow writes,
# or rows pinned by commit.
EVIDENCE = {
    name: entry for name, entry in (config.get("evidence") or {}).items() if entry
}


FLOW = PaperFlow.load(config, EVIDENCE, PAPER.parent, localconfig.expand)
# Every forecasting rule's device and threads (`forecast` in the workflow
# config, overridden by this machine's FORECAST_DEVICE/FORECAST_THREADS).
FORECAST = forecast.settings(config, localconfig.read())


R = FLOW.results


# The final run's final checkpoint, which every rule running an `_ft`
# model takes as an input and loads (`PaperFlow.finetuned_env`); empty
# while the fine-tuned models are pending.
FINETUNED = FLOW.finetuned_inputs()
FT_EXPORT = f"export {FLOW.finetuned_env()};" if FINETUNED else ""


BUILD = FLOW.build


STUDIES = config["studies"]


GERMANY = FLOW.germany


EBOLA = FLOW.ebola


SYNTHETIC = FLOW.synthetic


TRIANGLE = FLOW.triangle


EBOLA_DRAWS = FLOW.ebola_draws


# The tools the shell rules call, as the project workflow's.
TOOLS_ENV = "../../workflow/envs/tools.yaml"


# The branches a promote rule runs from (`promote_branches`).
PROMOTE_BRANCHES = " ".join(config["promote_branches"])


wildcard_constraints:
    mode="|".join(MODES),
    study="|".join(STUDIES),


# The release layer of the run the paper reports (`final_run`), if it has
# one: the grid table's last group.
def final_release_layer():
    run = Path(config["final_run"]).name if config.get("final_run") else None
    entry = (config.get("runs") or {}).get(run) or {}
    return entry.get(
        "release_layer", (config.get("finetune") or {}).get("release_layer")
    )
