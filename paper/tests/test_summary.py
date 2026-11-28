import pandas as pd
import yaml

from paperpipe.meta import DUMMY_LABEL, read_meta, write_meta
from paperpipe.outputs import to_markdown
from paperpipe.summary import finetuned_spec, study_summary, write_summary


def _scores(**extra) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["KIT-frozen_baseline", "a", "b"],
            "rel_score": [1.0, 0.8, 0.6],
            "rel_log_score": [1.0, 0.9, 0.7],
            **extra,
        }
    )


def test_summary_names_the_best_model_against_the_baseline() -> None:
    out = study_summary({"study": "germany", "source": "real"}, _scores())
    assert out["best_model"] == "b"
    assert out["best_rel_score"] == 0.6
    assert out["baseline"] == "KIT-frozen_baseline"
    assert out["label"] == ""


def test_summary_uses_the_heldout_half_and_labels_dummy_data() -> None:
    # Held-out rows first, as a table sorted by model may have them.
    scores = pd.concat(
        [
            _scores(period="held out").assign(rel_score=[1.0, 0.5, 0.9]),
            _scores(period="development"),
        ]
    )
    out = study_summary({"study": "germany", "source": "dummy"}, scores)
    assert out["best_model"] == "a"
    assert out["label"] == DUMMY_LABEL


def test_meta_and_summary_round_trip(tmp_path) -> None:
    write_meta(tmp_path / "m.json", "ebola", "dummy")
    assert read_meta(tmp_path / "m.json")["source"] == "dummy"
    write_summary(tmp_path / "r.yml", "development", {"ebola": {"n": 1}})
    doc = yaml.safe_load((tmp_path / "r.yml").read_text())
    assert doc["results"]["mode"] == "development"
    assert doc["results"]["ebola"] == {"n": 1}


def test_markdown_tables_round_numbers_and_escape_pipes() -> None:
    md = to_markdown(pd.DataFrame({"model": ["a|b"], "x": [0.123456]}))
    lines = md.splitlines()
    assert lines[0] == "| model | x |"
    assert lines[2] == "| a\\|b | 0.123 |"


def test_markdown_tables_leave_missing_models_blank() -> None:
    md = to_markdown(pd.DataFrame({"model": ["randomwalk", None], "x": [1, 2]}))
    lines = md.splitlines()
    assert lines[2].startswith("| Random walk |")
    assert "nan" not in md.lower()


def test_summary_says_when_the_scored_part_has_no_scores_yet() -> None:
    scores = _scores(period="development")
    out = study_summary({"study": "ebola", "source": "real"}, scores)
    assert out["best_model"] == ""
    assert out["n_models"] == 0
    assert "no scored forecasts yet" in out["status"].lower()
    scored = study_summary({"study": "germany", "source": "real"}, _scores())
    assert scored["status"] == ""


def test_summary_says_which_part_its_numbers_come_from() -> None:
    germany = {"study": "germany", "source": "real"}
    ebola = {"study": "ebola", "source": "real"}
    assert study_summary(germany, _scores())["scope"] == "development half"
    held = _scores(period="held out")
    assert study_summary(germany, held)["scope"] == "held-out half"
    assert study_summary(ebola, _scores())["scope"] == (
        "release backtest before the scored rounds"
    )
    assert study_summary(ebola, held)["scope"] == "scored rounds"
    synthetic = {"study": "synthetic", "source": "real"}
    assert study_summary(synthetic, _scores(), mode="development")["scope"] == (
        "development simulations"
    )
    assert study_summary(synthetic, _scores(), mode="paper")["scope"] == (
        "held-out simulations"
    )


def test_summary_records_the_fine_tuned_checkpoint(tmp_path) -> None:
    path = tmp_path / "results.yml"
    write_summary(path, "paper", {}, finetuned="e8a1f5ae338b")
    doc = yaml.safe_load(path.read_text())["results"]
    assert doc["finetuned"] == "e8a1f5ae338b"
    write_summary(path, "paper", {})
    assert yaml.safe_load(path.read_text())["results"]["finetuned"] == "pending"


def test_summary_marks_each_study_it_has_as_shown() -> None:
    # The paper hides a study's results when its run has no entry for it.
    out = study_summary({"study": "germany", "source": "real"}, _scores())
    assert out["shown"] is True


def test_summary_quotes_each_model_against_randomwalknowcast() -> None:
    scores = pd.DataFrame(
        {
            "model": [
                "KIT-frozen_baseline",
                "randomwalknowcast",
                "foundation_zeroshot_baselinenowcast",
            ],
            "rel_score": [1.0, 1.2, 0.9],
            "rel_log_score": [1.0, 1.1, 0.95],
            "rel_score_randomwalknowcast": [0.83, 1.0, 0.754],
        }
    )
    out = study_summary({"study": "germany", "source": "real"}, scores)
    assert out["models"]["foundation_zeroshot_baselinenowcast"] == 0.75
    # A quoted model without scores yet reads as pending.
    assert out["models"]["foundation_baselinenowcast"] == "pending"


def test_with_randomwalknowcast_the_baseline_its_scores_are_quoted() -> None:
    scores = pd.DataFrame(
        {
            "model": ["randomwalknowcast", "foundation"],
            "rel_score": [1.0, 0.8],
            "rel_log_score": [1.0, 0.9],
        }
    )
    out = study_summary({"study": "synthetic", "source": "dummy"}, scores)
    assert out["models"]["foundation"] == 0.8


def test_every_quoted_model_score_is_one_the_summary_writes() -> None:
    import re
    from pathlib import Path

    from paperpipe.summary import QUOTED_MODELS

    here = Path(__file__).parents[1]
    for name in ("main.qmd", "si.qmd"):
        text = (here / name).read_text()
        for model in re.findall(r"results\.\w+\.models\.(\w+) >", text):
            assert model in QUOTED_MODELS, (name, model)


def test_the_checkpoint_spec_is_read_beside_its_release_record(
    tmp_path,
) -> None:
    segment = tmp_path / "n3200"
    segment.mkdir()
    (segment / "settings.json").write_text(
        '{"hash": "e8a1f5ae338b", "series_bias": true}'
    )
    (segment / "published.json").write_text('{"tag": "t", "hash": "x"}')
    assert finetuned_spec(segment / "published.json") == "slots-e8a1f5ae338b"


def test_summary_gives_the_training_checkpoints_from_the_config(
    tmp_path,
) -> None:
    path = tmp_path / "results.yml"
    write_summary(
        path, "paper", {}, prefixes=[100, 200, 400, 800, 1600, 3200, 4800]
    )
    training = yaml.safe_load(path.read_text())["results"]["training"]
    assert training == {
        "simulations": "4,800",
        "checkpoints": "100, 200, 400, 800, 1,600, 3,200 and 4,800",
        "first": "100",
        "first_rate": "98%",
        "penultimate": "3,200",
        "penultimate_rate": "33%",
    }


def test_summary_quotes_each_model_against_the_study_s_baseline() -> None:
    scores = pd.DataFrame(
        {
            "model": ["KIT-frozen_baseline", "randomwalknowcast", "foundation"],
            "rel_score": [1.0, 1.2, 0.9],
            "rel_log_score": [1.0, 1.1, 0.95],
            "rel_score_randomwalknowcast": [0.83, 1.0, 0.75],
        }
    )
    out = study_summary({"study": "germany", "source": "real"}, scores)
    assert out["vs_baseline"]["foundation"] == 0.9
    assert out["vs_baseline"]["randomwalknowcast"] == 1.2
    assert out["vs_baseline"]["foundation_zeroshot"] == "pending"


def test_summary_ranks_models_by_pairwise_skill() -> None:
    scores = pd.DataFrame(
        {
            "model": [
                "BDBVHub-baseline",
                "foundation",
                "seabbs-renewal",
                "randomwalknowcast",
                "no-pairwise",
            ],
            "rel_score": [1.0, 0.7, 0.9, 1.1, 0.5],
            "rel_log_score": [1.0, 0.8, 0.9, 1.0, 0.5],
            "pair_score": [1.0, 0.6, 0.95, 0.8, float("nan")],
        }
    )
    out = study_summary({"study": "ebola", "source": "real"}, scores)
    assert out["rank"] == {
        "n": 4,
        "foundation": 1,
        "randomwalknowcast": 2,
        "seabbs_renewal": 3,
    }


def test_a_study_a_run_leaves_out_is_pending_and_hidden(tmp_path) -> None:
    path = tmp_path / "r.yml"
    write_summary(path, "development", {"germany": {"n": 1}})
    doc = yaml.safe_load(path.read_text())["results"]
    assert doc["germany"] == {"n": 1}
    ebola = doc["ebola"]
    assert ebola["shown"] is False
    assert ebola["rank"]["foundation"] == "pending"
    assert ebola["vs_baseline"]["randomwalknowcast"] == "pending"


def _paths(text: str, root: str) -> set[tuple[str, ...]]:
    import re

    return {
        tuple(m.split("."))
        for m in re.findall(rf"meta {root}\.([\w.]+) >", text)
    }


def _resolve(doc: dict, path: tuple[str, ...]) -> bool:
    node = doc
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def test_every_results_key_the_paper_quotes_is_written(tmp_path) -> None:
    from pathlib import Path

    here = Path(__file__).parents[1]
    text = "".join((here / n).read_text() for n in ("main.qmd", "si.qmd"))
    scores = pd.DataFrame(
        {
            "model": ["randomwalknowcast", "foundation"],
            "rel_score": [1.0, 0.8],
            "rel_log_score": [1.0, 0.9],
            "pair_score": [1.0, 0.8],
        }
    )
    studies = {
        s: study_summary({"study": s, "source": "dummy"}, scores, "paper")
        for s in ("synthetic", "germany")
    }
    path = tmp_path / "r.yml"
    write_summary(path, "development", studies, prefixes=[100, 200])
    doc = yaml.safe_load(path.read_text())["results"]
    missing = [p for p in _paths(text, "results") if not _resolve(doc, p)]
    assert not missing, missing
