"""Scoring units' commands and the German report's arguments."""

from flowlib import scoring


def test_a_unit_waits_for_room_then_scores_and_stamps():
    got = scoring.unit_shell("germany", "wait && ", "{params.segment} ")
    assert got.startswith("mkdir -p {params.copy_back} && wait && ")
    assert (
        f"{scoring.EVALUATE} germany {{params.cache}} {{params.label}} " in got
    )
    assert got.endswith("{params.segment} > {log} 2>&1 && touch {output}")


def test_the_report_names_the_references_and_saved_checkpoints():
    got = scoring.report_args("runa", [100, 200], ["reference"], "/c/g")
    assert got == (
        "reference=Zero-shot 'randomwalknowcast=Random walk nowcast' "
        "'n100=runa n100' 'n200=runa n200' --from reference=/c/g/reference"
    )
