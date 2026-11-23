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


from paperpipe.flow import BDBV, PYTHON, PaperFlow  # noqa: E402

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


BUILD = FLOW.build


STUDIES = config["studies"]


GERMANY = FLOW.germany


EBOLA = FLOW.ebola


SYNTHETIC = FLOW.synthetic


TRIANGLE = FLOW.triangle


EBOLA_DRAWS = FLOW.ebola_draws


# The tools the shell rules call, as the project workflow's.
TOOLS_ENV = "../../workflow/envs/tools.yaml"


wildcard_constraints:
    mode="|".join(MODES),
    study="|".join(STUDIES),
