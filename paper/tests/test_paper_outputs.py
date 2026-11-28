"""Every figure and table the paper reads has a producer."""

import re
from pathlib import Path

from paperpipe.outputs import FIGURES, JOINT_STUDIES

PAPER = Path(__file__).parents[1]
TEXT = "\n".join((PAPER / n).read_text() for n in ("main.qmd", "si.qmd"))
STUDY_FIGURE = re.compile(
    r"build/figures/(main|si)/(synthetic|germany|ebola)-([\w-]+)\.png"
)


def test_every_study_figure_the_paper_reads_is_made() -> None:
    read = set(STUDY_FIGURE.findall(TEXT))
    made = {
        (part, study, name)
        for study, parts in FIGURES.items()
        for part, names in parts.items()
        for name in names
    }
    assert read <= made, sorted(read - made)


def test_every_study_figure_made_is_read() -> None:
    read = set(STUDY_FIGURE.findall(TEXT))
    made = {
        (part, study, name)
        for study, parts in FIGURES.items()
        for part, names in parts.items()
        for name in names
    }
    assert made <= read, sorted(made - read)


def test_the_joint_tables_the_paper_reads_are_the_joint_studies() -> None:
    read = set(re.findall(r"build/tables/main/(\w+)-joint\.md", TEXT))
    assert read == set(JOINT_STUDIES)


def test_no_figure_needs_checkpoint_forecasts_of_the_synthetic_study() -> None:
    # The synthetic runner forecasts the final checkpoint only, so no
    # figure plots scores by fine-tuning outbreaks (`<model>@<n>` rows).
    assert "si-finetune-size" not in FIGURES["synthetic"]["si"]


def test_trajectories_are_scored_jointly_for_the_ebola_hub_only() -> None:
    assert JOINT_STUDIES == ("ebola",)
