"""The dashboard page: training status, losses, bias, scores and nowcasts.

`build` returns the whole page as HTML, in sections behind a sticky bar
of section links, the live run's step and one shared run and reference
selector that every chart follows. Its data come from the runs' folders
(`logs`, `training`, `bias`), the score table (`scores`), each
configured dataset's forecasts (`nowcasts`) and an optional mailbox,
through `sources.Sources`; its runs, references, datasets and defaults
from configuration. The page's
scripts, style and tooltips are in `page/`.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nowcasteval import labels

from . import (
    bias,
    kv,
    level_loss_lora,
    logs,
    nowcasts,
    paperstyle,
    scores,
    svg,
    training,
)
from .logs import _size
from .runs import Reference, Run, checkpoint_settings, split_label
from .sources import Sources

ASSETS = Path(__file__).with_name("page")
MODES = ("raw", "start", "base")
STALE = 600
"""Seconds after which the last log line is stale."""
BIAS_PANEL = (320, 180)
"""Every attention bias panel's size (width, height), the learning rate's
too, so the section is an even grid (page/bias.js draws the rest)."""
ADAPTER = "finetuned-ckpt/adapter_model.safetensors"
"""A checkpoint's adapter, which the trainer writes before it scores it."""


def _asset(name: str) -> str:
    return (ASSETS / name).read_text()


EXPLAIN: dict[str, str] = json.loads(_asset("explain.json"))
"""Plain-language notes behind each info marker."""


@dataclass
class Context:
    """What one page shows and where its data come from."""

    runs: Sequence[Run]
    references: Sequence[Reference]
    source: Sources
    rule: str
    outbreaks: int
    learning_rate: float
    prefixes: Sequence[int]  # `finetune.prefixes`
    datasets: Sequence[Mapping[str, Any]] = ()
    mailbox: Path | None = None
    mail_filter: str = ""
    mail_hide: str = ""
    hidden_studies: set = field(default_factory=set)
    mail_lines: int = 400
    once: bool = False
    view: Mapping[str, Any] = field(default_factory=dict)
    logs: dict[str, logs.RunLog] = field(default_factory=dict)
    stores: dict[str, dict] = field(default_factory=dict)
    excluded: set = field(default_factory=set)
    held_out_from: dict[str, str] = field(default_factory=dict)
    """Each dataset's held-out half's first nowcast date: never read."""

    def _runlog(self, run: Run) -> logs.RunLog:
        key = str(run.folder)
        if key not in self.logs:
            self.logs[key] = logs.RunLog(run.folder)
        return self.logs[key]

    def log(self, run: Run) -> list[dict[str, Any]]:
        """Return a run's logged steps (none for a run with no folder)."""
        if run.folder is None:
            return []
        return self._runlog(run).rows()

    def sims(self, run: Run) -> Callable[[float], float]:
        """Return the simulations a run has trained by a step.

        From its own checkpoints and logged chunks (`training.anchors`),
        for the curves' simulations axis; where the run is now comes from
        its status alone (`live_status`).
        """
        pairs, spo = self._anchors(run)
        return lambda step: training.sims_at(pairs, step, spo)

    def _anchors(self, run: Run) -> tuple[list[tuple[int, int]], float]:
        cks = _checkpoints(run)
        chunks = self._runlog(run).chunks() if run.folder else {}
        pairs = training.anchors(cks, chunks)
        return pairs, training.steps_per_simulation(cks)

    def dataset(self, key: str) -> Mapping[str, Any]:
        """Return a configured dataset by key (an empty one if unknown)."""
        return next(
            (d for d in self.datasets if d["key"] == key),
            {"key": key, "label": key, "groups": []},
        )

    def nowcasts(self, key: str) -> dict[str, Any] | None:
        """Return a dataset's nowcasts, rebuilt only when its files change."""
        ds = self.dataset(key)
        days, ranges, _ = nowcasts.horizons(ds)
        return nowcasts.cached_build(
            self.source.dataset(key),
            self.runs,
            self.references,
            self.stores.setdefault(key, {}),
            levels=nowcasts.LEVELS,
            excluded=self.excluded,
            order=[(g["name"], g["match"]) for g in ds.get("groups") or ()],
            sources=ds.get("sources"),
            unit=ds.get("unit") or key,
            days=days,
            ranges=ranges,
            until=self.held_out_from.get(key),
        )


def _json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def ui_colours() -> str:
    """Return the dashboard's accent colours as CSS variables.

    From the shared map (`labels.UI_COLOURS`), light and dark; the page's
    stylesheet holds only neutrals.
    """
    light = ";".join(f"--{k}:{v[0]}" for k, v in labels.UI_COLOURS.items())
    dark = ";".join(f"--{k}:{v[1]}" for k, v in labels.UI_COLOURS.items())
    return (
        f":root{{{light}}}@media(prefers-color-scheme:dark){{:root{{{dark}}}}}"
    )


def tip(key: str) -> str:
    """Return the info marker of a note in `EXPLAIN`."""
    text = html.escape(EXPLAIN.get(key, ""), quote=True)
    return f'<span class=info tabindex=0 data-tip="{text}">ⓘ</span>'


def _fmt(v: Any, n: int = 4) -> str:
    if v is None:
        return ""
    return f"{v:.{n}f}" if isinstance(v, float) else html.escape(str(v))


def _dur(s: float | None) -> str:
    if s is None or s != s or s < 0:
        return "n/a"
    return kv.duration(s)


def _checkpoints(run: Run) -> list[tuple[str, dict, float | None]]:
    return [] if run.folder is None else logs.checkpoints(run.folder)


@dataclass
class Status:
    """The live run's state, for the tiles and the sticky bar.

    `step` and `trained` (simulations) are the trainer's status
    (`training.status_position`), None without one; `rate` is the steps
    a second since the last restart.
    """

    step: int | None
    rate: float | None
    last_t: float | None
    params: dict[str, Any]
    saved: set[int]
    pending: list[int]
    nxt: int | None
    trained: float | None = None
    span: float | None = None
    mean_rate: float | None = None

    @property
    def target(self) -> int:
        """Return the final checkpoint's size."""
        return max(self.params["prefixes"] or [0])

    def eta(self, n: int | None) -> str:
        """Return the time until checkpoint `n` at the rate since restart.

        The steps still to go are the simulations still to go at the
        run's own steps a simulation so far (`step / trained`).
        """
        if not n or self.step is None or not self.trained:
            return "n/a"
        if self.trained >= n:
            return "due now"
        if not self.rate:
            return "n/a"
        spo = self.step / self.trained
        return _dur((n - self.trained) * spo / self.rate)


def run_sizes(prefixes: Sequence[int], outbreaks: int) -> list[int]:
    """Return a run's checkpoint sizes as the trainer makes them.

    The prefixes below the run's simulations, then the run's simulations.
    """
    return sorted({int(p) for p in prefixes if p < outbreaks} | {outbreaks})


def live_status(
    live: Run, rows: list[dict[str, Any]], planned: Sequence[int]
) -> Status:
    """Return the live run's position, speed and checkpoints.

    `planned` are its checkpoints, with those saved (`logs.prefixes`).
    The step and simulations trained are the trainer's status alone
    (`training.status_position`); the logged steps since the last
    restart (`training.resumed_at`) and the status give the rate.
    """
    folder = live.folder
    last_t = rows[-1]["t"] if rows else None
    position = training.status_position(folder) if folder else None
    step = position["step"] if position else None
    trained = float(position["outbreaks"]) if position else None
    # Timestamped steps: every logged step, and the trainer's status file
    # (written every few steps), as evaluations log a step only every few
    # hundred (`training.recent_rate`).
    points = [(r["t"], r["step"]) for r in rows]
    status = training.status_point(folder) if folder else None
    if status:
        points.append(status)
    since = training.resumed_at(folder) if folder else None
    rate, span = training.recent_rate(points, since=since)
    if since is not None and not any(t >= since for t, _ in points):
        rate, span = None, None
    params = (
        training.run_params(
            folder, training.history(live), step, planned, trained
        )
        if folder
        else {
            "lr_outbreaks": 0,
            "groups": {},
            "current": {},
            "logged": {},
            "prefixes": list(planned),
        }
    )
    sizes = params["prefixes"]
    saved = {
        n
        for n in sizes
        if folder and (folder / f"train/n{n}/settings.json").exists()
    }
    # Written but not yet scored at the checkpoint: its adapter is saved.
    # Snakemake makes a segment's empty output folders when it starts.
    pending = [
        n
        for n in sizes
        if n not in saved
        and folder
        and (folder / f"train/n{n}/{ADAPTER}").exists()
    ]
    nxt = next((n for n in sizes if n not in saved and n not in pending), None)
    return Status(
        step,
        rate,
        last_t,
        params,
        saved,
        pending,
        nxt,
        trained,
        span,
        training.mean_rate(points),
    )


def _rates(r: Mapping[str, Any]) -> str:
    # Train is the GPU's share less evaluation; waiting shows only when
    # it takes some time.
    sh = r["share"]
    split = [f"cut {sh['cut']:.0%}", f"eval {sh['eval']:.0%}"]
    if sh["wait"] >= 0.005:
        split.append(f"wait {sh['wait']:.0%}")
    return (
        f"{r['steps_per_min']:.1f} steps/min, "
        f"{r['sims_per_hour']:.0f} simulations/h, "
        f"GPU busy {r['gpu_busy']:.0%} ({', '.join(split)})"
    )


def throughput_line(folder: Path | None, segment: str | None) -> str:
    """Return one line of throughput: the running segment and the run.

    From the trainer's chunk-time lines (`logs.throughput`); empty for a
    run with no finished chunk.
    """
    if folder is None:
        return ""
    rows = logs.RunLog(folder).chunk_times()
    run = logs.throughput(rows)
    if run is None:
        return ""
    parts = []
    seg = logs.throughput([r for r in rows if r["segment"] == segment])
    if seg:
        n = seg["chunks"]
        parts.append(
            f"this segment ({html.escape(str(segment))}, {n} chunk"
            f"{'s' if n != 1 else ''}): {_rates(seg)}"
        )
    parts.append(f"whole run ({run['chunks']} chunks): {_rates(run)}")
    return (
        f"<p class=tput><span class=k>Throughput{tip('t_tput')}</span> "
        f"{' · '.join(parts)}</p>"
    )


def training_now(
    folder: Path | None, status: Mapping[str, Any], rate: float | None
) -> str:
    """Return one line of what the trainer is doing now.

    The status file's stage and, while a chunk trains, its simulations,
    the steps done of it (the step less the finished chunks' steps) and
    the time left at `rate` steps a second. Empty with no status.
    """
    stage = status.get("stage")
    if not stage:
        return ""
    parts = [html.escape(str(stage))]
    seg, step = status.get("segment"), status.get("step")
    if folder is not None and seg and step is not None:
        run = logs.RunLog(folder)
        done = run.chunk_times()
        finished = {r["first"] for r in done}
        open_ = [
            (a + 1, b, n)
            for a, (b, n) in run.chunks().items()
            if a + 1 not in finished
        ]
        if open_:
            a, b, n = max(open_)
            into = step - sum(r["steps"] for r in done)
            if 0 <= into <= n:
                parts.append(f"simulations {a}-{b} of {html.escape(seg)}")
                parts.append(f"{into:,} of {n:,} steps")
                if rate:
                    parts.append(f"{_dur((n - into) / rate)} left")
    return (
        f"<p class=tnow><span class=k>Now{tip('t_now')}</span> "
        f"{', '.join(parts)}</p>"
    )


def checkpoint_timings(folder: Path | None) -> str:
    """Return a table of each checkpoint's chunks, steps and time split.

    From the trainer's chunk-time lines (`logs.RunLog.chunk_times`); a
    segment with no saved checkpoint yet is marked as running so far.
    """
    if folder is None:
        return ""
    rows = logs.RunLog(folder).chunk_times()
    if not rows:
        return ""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["segment"], []).append(r)
    head = (
        "<tr><th>checkpoint</th><th>chunks</th><th>steps</th>"
        "<th>training</th><th>cutting</th><th>evaluating</th>"
        "<th>in all</th><th>last chunk ended</th></tr>"
    )
    body = []
    for seg in sorted(by, key=logs._size):
        rs = by[seg]
        part = {k: sum(r[k] for r in rs) for k in logs.SPLIT}
        ended = datetime.fromtimestamp(max(r["t"] for r in rs))
        saved = (folder / "train" / seg / "settings.json").exists()
        body.append(
            f"<tr><td>{html.escape(seg)}</td><td>{len(rs)}</td>"
            f"<td>{sum(r['steps'] for r in rs):,}</td>"
            f"<td>{_dur(part['train'])}</td><td>{_dur(part['cut'])}</td>"
            f"<td>{_dur(part['eval'])}</td>"
            f"<td>{_dur(sum(part.values()))}</td>"
            f"<td>{ended:%d %b %H:%M}{'' if saved else ' (running)'}</td></tr>"
        )
    return (
        "<details class=about open><summary class=mut>Checkpoint timings"
        f"{tip('t_timings')}</summary><div class=wrap><table class=tim>"
        f"{head}{''.join(body)}</table></div></details>"
    )


def status_tiles(st: Status, live: Run) -> tuple[str, str]:
    """Return the status tiles and the sticky bar's step and next checkpoint."""
    now = datetime.now().timestamp()
    folder = live.folder
    trained = st.trained
    trainer, alive = logs.trainer_state(folder) if folder else ("n/a", None)
    # A stopped run's next segment may wait on batches it lacks (failed or
    # still being simulated): name them.
    missing: list[int] = []
    if folder and not alive and st.nxt and not st.pending:
        missing = logs.waiting_batches(folder, st.nxt)
        if missing:
            alive = False
    # The running segment's last line of any kind, not only a step.
    last = logs.RunLog(folder).last_line() if folder else None
    last_t, last_text = last or (st.last_t, "")
    age = now - last_t if last_t else None
    stale = age is not None and age > STALE
    # The next prefix, the simulations done of it and its ETA. A written
    # checkpoint still being scored is named as itself.
    next_v = f"n{st.nxt}" if st.nxt else "none"
    eta = f"ETA {st.eta(st.nxt)}" if st.nxt else ""
    next_s: dict[str, Any] = {}
    if st.nxt:
        next_s = {
            "simulations": f"{min(trained, st.nxt):.0f}/{st.nxt}"
            if trained is not None
            else None,
            "ETA": st.eta(st.nxt),
        }
    if st.pending:
        next_s["scoring"] = f"n{st.pending[0]}"
        next_s["since"] = datetime.fromtimestamp(
            (folder / f"train/n{st.pending[0]}/{ADAPTER}").stat().st_mtime
        ).strftime("%H:%M")
    p = st.params
    rates = "\n".join(f"{g} {v:.2e}" for g, v in p["current"].items()) or "n/a"
    if p["lr_outbreaks"]:
        decay = f"linear decay to 0 at {p['lr_outbreaks']} simulations"
    else:
        decay = "constant"
    decay += ", from history" if p["logged"] else ""
    tiles = [
        (
            "Step",
            "t_step",
            "n/a" if st.step is None else str(st.step),
            {
                "simulations_trained": f"{trained:.0f}/{st.target}"
                if trained is not None
                else None,
            },
            "",
        ),
        (
            "Steps / s",
            "t_rate",
            f"{st.rate:.3f}" if st.rate else "n/a",
            {
                "mean": st.mean_rate,
                "window_s": st.span,
            },
            "",
        ),
        ("Next checkpoint", "t_next", next_v, next_s, ""),
        (
            "Segments",
            "t_segments",
            next_v,
            {
                f"n{n}": st.eta(n)
                for n in st.params["prefixes"]
                if n not in st.saved and n not in st.pending
            },
            "",
        ),
        (
            f"ETA n{st.target}",
            "t_eta",
            st.eta(st.target),
            {},
            "",
        ),
        ("Learning rates", "t_lr", rates, {"schedule": decay}, ""),
        (
            "Trainer",
            "t_trainer",
            trainer,
            {"waiting_for_batches": missing},
            "" if alive is None else " ok" if alive else " bad",
        ),
        (
            "Last log line",
            "t_log",
            datetime.fromtimestamp(last_t).strftime("%H:%M:%S")
            if last_t
            else "n/a",
            {
                "stale": stale or None,
                "age_s": age,
                "line": last_text[:80],
            },
            " bad" if stale else "",
        ),
    ]
    out = "".join(
        f'<div class="tile{cls}"><div class=k>{k}{tip(key)}</div>'
        f'<div class="v{" vl" if chr(10) in v else ""}">'
        f"{html.escape(v).replace(chr(10), '<br>')}</div>"
        f"<div class=s>{kv.box(s)}</div></div>"
        for k, key, v, s, cls in tiles
    )
    done = f"n{trained:.0f} / " if trained is not None and st.nxt else ""
    bar = (
        f"<span class=bt><span class=k>step</span> <b>{st.step}</b> "
        f"<span class=k>next</span> <b>{html.escape(done + next_v)}</b>"
        f"{' ' + html.escape(eta) if eta else ''}</span>"
        if st.step is not None
        else ""
    )
    return out, bar


def lr_card(
    params: Mapping[str, Any],
    step: int | None,
    run: str,
    sims: Callable[[float], float] | None = None,
) -> str:
    """Return a chart of every group's learning rate.

    Against simulations trained for a run whose rate falls over its
    simulations (`sims`, `Context.sims`), else against step.
    """
    if not params["groups"]:
        return ""
    by_sims = bool(params["lr_outbreaks"] and sims)
    width, height = BIAS_PANEL
    left, right, top, bottom = 40, 8, 6, 30
    decays = params["lr_outbreaks"] if by_sims else 0

    def x_of(s: float) -> float:
        return sims(s) if by_sims and sims else s

    now = x_of(step) if step is not None else None
    xmax = decays or max([now or 0, 1])
    vals = [max(b, 1e-12) for b in params["groups"].values()]
    # A linear decay reaches zero, so the log axis stops at 1% of the base.
    lo = math.log10(min(vals) * (0.01 if decays else 1)) - 0.1
    hi = math.log10(max(vals)) + 0.15

    def sx(x: float) -> float:
        return left + x / xmax * (width - left - right)

    def sy(v: float) -> float:
        frac = (math.log10(max(v, 1e-12)) - lo) / (hi - lo)
        return top + (1 - frac) * (height - top - bottom)

    o = [
        f'<svg class=srv viewBox="0 0 {width} {height}" width="100%" '
        'role="img">'
    ]
    for e in range(math.ceil(lo), math.floor(hi) + 1):
        o.append(
            f'<line x1="{left}" x2="{width - right}" y1="{sy(10**e):.1f}" '
            f'y2="{sy(10**e):.1f}" class=grid /><text x="{left - 3}" '
            f'y="{sy(10**e) + 3:.1f}" text-anchor=end class=ax>1e{e}</text>'
        )
    for t in svg.nice(0, xmax, 4):
        if t <= xmax:
            o.append(
                f'<text x="{sx(t):.1f}" y="{height - bottom + 14}" '
                f"text-anchor=middle class=ax>{t:g}</text>"
            )
    o.append(
        f'<text x="{(left + width - right) / 2}" y="{height - 2}" '
        f"text-anchor=middle class=ax>"
        f"{'simulations trained' if by_sims else 'step'}</text>"
    )
    legend = []
    for i, (g, base) in enumerate(params["groups"].items()):
        col = svg.PALETTE[i % len(svg.PALETTE)]
        logged = [(x_of(s), v) for s, v in params["logged"].get(g, [])]
        pts = logged or (
            [(0, base), (0.99 * xmax, base * 0.01)]
            if decays
            else [(0, base), (xmax, base)]
        )
        o.append(
            '<polyline points="'
            + " ".join(f"{sx(x):.1f},{sy(max(y, 1e-12)):.1f}" for x, y in pts)
            + f'" fill=none stroke="{col}" stroke-width=1.6 />'
        )
        cur = params["current"].get(g)
        if cur is not None and now is not None:
            at = f"{now:.0f} simulations" if by_sims else f"step {step}"
            o.append(
                f'<circle cx="{sx(now):.1f}" cy="{sy(cur):.1f}" r=3 '
                f'fill="{col}"><title>{html.escape(g)} {cur:.3e} at '
                f"{at}</title></circle>"
            )
        legend.append(
            f'<span><i style="background:{col}"></i>{html.escape(g)}</span>'
        )
    o.append("</svg>")
    return (
        f'<div class=card id=lrcard data-run="{html.escape(run, quote=True)}">'
        f"<h3>Learning rate vs "
        f"{'simulations trained' if by_sims else 'step'}{tip('lr')}</h3>"
        f"<div class=leg>{''.join(legend)}</div>{''.join(o)}</div>"
    )


LOSS_TYPES = {
    "total": "total",
    "nowcast": "nowcast",
    "forecast": "forecast",
    "layout": "by layout",
    "level": "by level",
}
"""The validation loss types the Loss panel offers."""


def loss_section(ctx: Context, live: Run, st: Status) -> str:
    """Return the loss controls and the client-side loss plots' data."""
    cks = {r.key: _checkpoints(r) for r in ctx.runs}
    out = [
        training.loss_run(
            r, ctx.log(r), cks[r.key], ctx.sims(r), None, r.colour
        )
        for r in ctx.runs
    ]
    # x is simulations trained: checkpoints sit at their own sizes.
    marks = [
        (int(s.get("n_outbreaks") or _size(nm)), nm)
        for nm, s, _ in cks[live.key]
    ] + [(n, f"n{n}?") for n in st.pending]
    now = ctx.sims(live)(st.step) if st.step else 0
    next_ck = next(
        (n for n in st.params["prefixes"] if n not in st.saved), None
    )
    xlim = max([1.2 * now, 50] + ([next_ck] if next_ck else []))
    # Layout and level colours from the shared map; others by position.
    data = _json(
        {
            "runs": out,
            "marks": marks,
            "xlim": xlim,
            "layouts": labels.LAYOUT_COLOURS,
            "levels": labels.LEVEL_COLOURS,
            "categorical": list(labels.CATEGORICAL),
        }
    )
    modebar = (
        '<div class="sctl modebar"><span class=mut>Loss scale</span>'
        "<button class=tog data-mode=raw>raw</button>"
        "<button class=tog data-mode=start title=\"divided by the run's own "
        'step-0 loss">relative to start</button>'
        '<button class=tog data-mode=base title="divided by base Chronos-2 on '
        "the same blocks (a run's base)\">relative to base Chronos-2</button>"
        "<span class=sep></span><span class=mut>x range</span>"
        "<button class=tog data-rng=zoom>live run</button>"
        "<button class=tog data-rng=full>full range</button>"
        + tip("lossscale")
        + "</div>"
        # Which held-out loss the held-out panel draws, for every run.
        + '<div class="sctl modebar"><span class=mut>Loss type</span>'
        + "".join(
            f"<button class=tog data-lt={k}>{v}</button>"
            for k, v in LOSS_TYPES.items()
        )
        + tip("losstype")
        + "</div>"
    )
    return (
        f"{modebar}<div id=lossrefs class=refnote></div>"
        "<div class=plots3 id=lossplots></div>"
        f'<script type="application/json" id=ldata>{data}</script>'
        f"<script>{_asset('axes.js')}\n{_asset('loss.js')}</script>"
    )


def bias_section(ctx: Context, live: Run, st: Status) -> str:
    """Return the attention bias panels and the live run's learning rates.

    One even grid of equal panels (`BIAS_PANEL`): each relation's bias
    over training then the learning rate, and each relation's latest
    matrix then the mean by layer; page/bias.js draws them around the
    learning rate's card, with one legend above (`biashead`).
    """
    data = _json(
        bias.payload(
            ctx.runs,
            st.step,
            {r.key: ctx.log(r) for r in ctx.runs},
            {r.key: ctx.sims(r) for r in ctx.runs},
        )
    )
    return (
        "<div id=biasrefs class=refnote></div><div id=biashead "
        "class=biashead></div><div class=biasgrid id=biasgrid>"
        f"{lr_card(st.params, st.step, live.key, ctx.sims(live))}</div>"
        f'<script type="application/json" id=bdata>{data}</script>'
        f"<script>{_asset('bias.js')}</script>"
    )


def seven_day_check(
    s: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None]:
    """Return a checkpoint's 7-day check (rule seven_day_check).

    The 7-day row's median over the 7-day sums of the daily rows'
    medians, the share of those sums inside its 90% interval, and the
    roughness of its median over the truth's (1 is as smooth as the
    truth; more carries a weekday cycle). None where not scored.
    """
    check = s.get("seven_day") or {}
    return check.get("ratio"), check.get("inside90"), check.get("rough_ratio")


def checkpoint_table(
    run: Run, info: Mapping[str, Any], mode: str, sources: Sequence[str] = ()
) -> str:
    """Return one run's checkpoint table at one loss scale.

    `sources` are the held-out sources given a column each.
    """
    rows = info["ck"]
    if not rows:
        label = html.escape(run.label)
        return f"<p class=mut>no checkpoints yet for {label}</p>"
    head = {
        "raw": "validation",
        "start": "validation / start",
        "base": "validation / base",
    }[mode]
    den = {"raw": 1, "start": info["full0"], "base": info["bfull"]}[mode]
    by_head = "".join(f"<th>{html.escape(src)}" for src in sources)
    out = [
        f"<table><tr><th>run<th>ckpt<th>hash<th>{head}<th>nowcast part"
        f"<th>forecast part<th>7-day / daily sums<th>sums in 7-day 90%"
        f"<th>7-day roughness / truth"
        f"<th>validation start{by_head}<th>saved</tr>"
    ]
    for name, s, mtime in rows:
        by = s.get("held_out_loss_by_source") or {}
        ho = s.get("held_out_loss")
        ratio, inside, rough = seven_day_check(s)
        cells = [
            html.escape(run.label),
            name,
            html.escape(str(s.get("hash", ""))),
            _fmt(ho / den if ho and den else None) or "n/a",
            # The held-out loss's two parts (labels.HORIZON_PARTS), at the
            # same scale as the loss.
            *(
                _fmt(part / den if part is not None and den else None) or "n/a"
                for part in (
                    s.get("held_out_loss_nowcast"),
                    s.get("held_out_loss_forecast"),
                )
            ),
            _fmt(ratio, 2) or "n/a",
            f"{inside:.0%}" if inside is not None else "n/a",
            _fmt(rough, 2) or "n/a",
            _fmt(s.get("held_out_loss_start")),
            *(_fmt(by.get(src)) or "n/a" for src in sources),
            datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M")
            if mtime
            else "",
        ]
        out.append("<tr>" + "".join(f"<td>{c}" for c in cells) + "</tr>")
    return "".join(out) + "</table>"


def checkpoint_sizes(
    ctx: Context, info: Mapping, score_rows: Sequence[Mapping[str, str]]
) -> dict[str, int]:
    """Return each checkpoint hash's size: saved settings, then scores."""
    sizes: dict[str, int] = {}
    for row in score_rows:
        key = split_label(row.get("checkpoint", ""))[0]
        try:
            n = int(float(row.get("outbreaks") or 0))
        except ValueError:
            n = 0
        if n and key:
            sizes.setdefault(key, n)
    for r in ctx.runs:
        for _, s, _ in info[r.key]["ck"]:
            if s.get("n_outbreaks"):
                sizes[str(s.get("hash"))] = int(s["n_outbreaks"])
        # Interim checkpoints, exported mid-segment, by their own hash.
        for path in checkpoint_settings(r.folder) if r.folder else ():
            try:
                s = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if s.get("interim") and s.get("n_outbreaks"):
                key = split_label(str(s.get("hash")))[0]
                sizes[key] = int(s["n_outbreaks"])
    return sizes


INPUT_LABELS = {"daily": "daily counts", "weekly": "weekly totals"}
"""The nowcast panels' model inputs, as the control names them."""


def nowcasts_section(
    ctx: Context, sizes: Mapping[str, int], builts: Mapping[str, Any]
) -> str:
    """Return the nowcast panels of every dataset, behind a selector.

    Every configured dataset is listed (Germany's development half; BDBV
    is held out). One with no nowcasts yet is listed as empty, so the
    selector always offers it.
    """
    datasets, sources = [], set()
    for ds in ctx.datasets:
        built = builts.get(ds["key"])
        if not built or not built["origins"] or not built["locs"]:
            datasets.append(
                {
                    "key": ds["key"],
                    "label": ds.get("label", ds["key"]),
                    "metric": ds.get("metric", ""),
                    "empty": True,
                    "truth": "",
                    "meta": [],
                    "locs": [],
                    "origins": [],
                    "targets": [],
                    "preload": {},
                    "groups": [],
                }
            )
            continue
        meta = nowcasts.meta(
            built, ctx.runs, ctx.references, sizes, ctx.excluded
        )
        days, ranges, default = nowcasts.horizons(ds)
        first = built["series"][built["locs"][0]]
        for m in meta:
            m["count"] = max(
                (
                    len(t)
                    for src in first.get(m["id"], {}).values()
                    for t in src.values()
                ),
                default=0,
            )
        for loc in built["series"].values():
            for by_target in loc.values():
                for by_source in by_target.values():
                    sources |= set(by_source)
        datasets.append(
            {
                "key": ds["key"],
                "label": ds.get("label", ds["key"]),
                "metric": ds.get("metric", ""),
                "truth": ds.get("truth", ""),
                "target": ds.get("target", ""),
                # The horizon control: every model is shown over `days`;
                # each choice is [name, first day, last day].
                "days": list(days) if days else None,
                "ranges": [[k, lo, hi] for k, (lo, hi) in ranges.items()],
                "range": default,
                "meta": meta,
                "locs": built["locs"],
                "origins": built["origins"],
                "targets": built["targets"],
                "preload": {
                    loc: nowcasts.location(built, loc) for loc in built["locs"]
                }
                if ctx.once
                else {},
                "groups": [
                    {
                        "name": g["name"],
                        "locs": [
                            x for x in built["locs"] if re.match(g["match"], x)
                        ],
                    }
                    for g in ds.get("groups") or ()
                ],
            }
        )
    if not any(not d.get("empty") for d in datasets):
        return "<p class=mut>No nowcasts yet.</p>"
    defaults = dict((ctx.view or {}).get("defaults") or {})
    data = _json(
        {
            "datasets": datasets,
            "runs": [
                {"key": r.key, "label": r.label, "color": r.colour}
                for r in ctx.runs
            ],
            "defaults": defaults,
            # WIS parts in the data's order, each with its fill (solid or
            # hatched, labels.WIS_PART_FILLS), and the order they stack in
            # (labels.WIS_PART_STACK, from the left).
            "parts": [
                [p, paperstyle.part_fill(p)] for p in paperstyle.WIS_PARTS
            ],
            "stack": list(paperstyle.WIS_STACK),
            # The horizon's parts: name, label and fill (labels).
            "horizon": [
                [k, labels.HORIZON_PARTS[k], labels.HORIZON_FILLS[k]]
                for k in labels.HORIZON_PARTS
            ],
        }
    )
    first = defaults.get("source")
    options = "".join(
        f'<option value="{html.escape(s, quote=True)}"'
        f"{' selected' if s == first else ''}>"
        f"{html.escape(INPUT_LABELS.get(s, s))}</option>"
        for s in sorted(sources)
    )
    controls = _asset("nowcasts.html").replace("{sources}", options)
    controls = controls.replace("{tip}", tip("nowcast"))
    return (
        controls + f'<script type="application/json" id=gdata>{data}</script>'
        f"<script>{_asset('bars.js')}\n{_asset('nowcasts.js')}</script>"
    )


def mail_section(ctx: Context) -> str:
    """Return the notes file's last matching lines; none without one."""
    if not ctx.mailbox:
        return ""
    if not ctx.mailbox.exists():
        return "<p class=mut>no notes yet</p>"
    rx = re.compile(ctx.mail_filter, re.I) if ctx.mail_filter else None
    # Lines about a held-out study (BDBV) are never shown.
    hide = re.compile(ctx.mail_hide, re.I) if ctx.mail_hide else None
    with ctx.mailbox.open(errors="replace") as f:
        tail = deque(f, maxlen=ctx.mail_lines)
    lines = [
        x.rstrip("\n")
        for x in tail
        if (not rx or rx.search(x)) and not (hide and hide.search(x))
    ]
    if not lines:
        return "<p class=mut>no matching lines</p>"

    def cut(s: str) -> str:
        return html.escape(s if len(s) <= 260 else s[:257] + "...")

    out = "<pre>" + "\n".join(cut(x) for x in lines[-5:]) + "</pre>"
    if len(lines) > 5:
        out += (
            f"<details><summary>show more ({len(lines) - 5})</summary><pre>"
            + html.escape("\n".join(lines[:-5]))
            + "</pre></details>"
        )
    return out


def tables_section(ctx: Context, info: Mapping[str, Any]) -> str:
    """Return the checkpoint tables and any configured tables (CSV files)."""
    sources = sorted(
        {
            src
            for r in ctx.runs
            for _, s, _ in info[r.key]["ck"]
            for src in (s.get("held_out_loss_by_source") or {})
        }
    )
    out = ["<h3>Checkpoints</h3><div class=wrap>"]
    out += [
        f'<div class="mode m-{m}">'
        + "".join(
            checkpoint_table(r, info[r.key], m, sources) for r in ctx.runs
        )
        + "</div>"
        for m in MODES
    ]
    out.append("</div>")
    return "".join(out)


def selector_data(
    ctx: Context,
    builts: Mapping[str, Any],
    sizes: Mapping[str, int] | None = None,
) -> str:
    """Return the shared run and reference selector's data.

    Other reference models and each checkpoint's two-step model are
    listed after the configured references, off until picked.
    """
    refs = [
        {"key": r.key, "label": r.label, "color": r.colour}
        for r in ctx.references
    ]
    for ds in ctx.datasets:
        built = builts.get(ds["key"])
        extra = (
            nowcasts.meta(built, ctx.runs, ctx.references, sizes or {})
            if built
            else []
        )
        for m in extra:
            s = m["id"]
            other = s.startswith(("model:", nowcasts.TWOSTEP))
            if other and not any(r["key"] == s for r in refs):
                refs.append(
                    {"key": s, "label": m["label"], "color": m["color"]}
                )
    shown = [r.key for r in ctx.runs] + [
        r.key for r in ctx.references if r.role in ("start", "baseline")
    ]
    return _json(
        {
            "runs": [
                {"key": r.key, "label": r.label, "color": r.colour}
                for r in ctx.runs
            ],
            "refs": refs,
            "defaults": shown,
        }
    )


def _parts(ctx: Context) -> tuple[Run, str, str, list]:
    """Return the live run, top bar status, selector data and sections.

    Each section is (id, title, content, open).

    Raises
    ------
    ValueError
        If no run is configured.
    """
    if not ctx.runs:
        raise ValueError("no runs to show: configure at least one run")
    live = next((r for r in ctx.runs if r.live), ctx.runs[0])
    rows = ctx.log(live)
    st = live_status(
        live,
        rows,
        run_sizes(ctx.prefixes, ctx.outbreaks),
    )
    tiles, bar = status_tiles(st, live)
    status = (
        logs.CACHE.get(live.folder / "train/stream-status.json") or {}
        if live.folder
        else {}
    )
    info = {
        r.key: training.loss_run(
            r, ctx.log(r), _checkpoints(r), ctx.sims(r), None, r.colour
        )
        | {"ck": _checkpoints(r)}
        for r in ctx.runs
    }
    for r in ctx.runs:
        info[r.key]["bfull"] = info[r.key]["full0"]
    score_rows = [
        r
        for r in scores.read(ctx.source.scores)
        if r.get("study") not in ctx.hidden_studies
    ]
    sizes = checkpoint_sizes(ctx, info, score_rows)
    mail = mail_section(ctx)
    builts = {ds["key"]: ctx.nowcasts(ds["key"]) for ds in ctx.datasets}
    views = training.views_summary(live.folder) if live.folder else None
    tput = throughput_line(live.folder, status.get("segment"))
    now = training_now(live.folder, status, st.rate)
    timings = checkpoint_timings(live.folder)
    about = {
        "trainer_status": status,
        "block_views": views,
        "updates_every_s": None if ctx.once else 60,
    }
    parts = [
        (
            "status",
            "Status",
            # The trainer's status, views and sources: folded, as the tiles
            # say where the run is.
            f"<div class=tiles>{tiles}</div>{now}{tput}{timings}"
            f"{also_training(ctx, live)}"
            "<details class=about>"
            f"<summary class=mut>Trainer status</summary>{kv.box(about)}"
            "</details>",
            True,
        ),
        ("loss", "Loss", loss_section(ctx, live, st), True),
        ("bias", "Attention bias", bias_section(ctx, live, st), True),
        (
            "lora",
            "LoRA update",
            level_loss_lora.section(ctx.runs, tip),
            True,
        ),
        ("nowcasts", "Nowcasts", nowcasts_section(ctx, sizes, builts), True),
    ]
    if mail:
        parts.append(("mailbox", "Recent activity", mail, True))
    tables = tables_section(ctx, info)
    parts.append(("tables", "Tables", tables, False))
    bar = f"{bar} <span class=mut>{datetime.now():%H:%M:%S}</span>"
    return live, bar, selector_data(ctx, builts, sizes), parts


def also_training(ctx: Context, live: Run) -> str:
    """Return the status tiles of each other run training now, if any.

    Runs train at once, a GPU each (flowlib.training): each run whose
    trainer is alive (`logs.trainer_state`) beside the live run.
    """
    out = []
    for r in ctx.runs:
        if r is live or not r.folder or not logs.trainer_state(r.folder)[1]:
            continue
        st = live_status(r, ctx.log(r), run_sizes(ctx.prefixes, ctx.outbreaks))
        tiles, _ = status_tiles(st, r)
        out.append(
            f"<h3><b>{html.escape(r.label)}</b></h3>"
            f"<div class=tiles>{tiles}</div>"
        )
    if not out:
        return ""
    return "<h3 class=mut>Also training</h3>" + "".join(out)


NAV = {
    "status": "Status",
    "mailbox": "Activity",
    "lora": "LoRA",
}
"""Short names in the top bar, where they differ from the section titles."""


def sections(ctx: Context) -> dict[str, str]:
    """Return the parts the page updates in place, by id.

    `bar` is the top bar's status; the rest are each section's content
    (`<details id=...>` without its summary).
    """
    return render(ctx)[1]


def build(ctx: Context) -> str:
    """Return the dashboard page.

    Raises
    ------
    ValueError
        If no run is configured.
    """
    return render(ctx)[0]


def render(ctx: Context) -> tuple[str, dict[str, str]]:
    """Return the dashboard page and its in-place parts, from one build.

    The parts are `sections`'; the page is `build`'s.
    """
    live, bar, seldata, parts = _parts(ctx)
    updates = {"bar": bar} | {key: content for key, _, content, _ in parts}
    nav = "".join(
        f'<a href="#{key}">{NAV.get(key, title)}</a>'
        for key, title, _, _ in parts
    )
    rest = "\n".join(
        f"<details class=sec id={key}{' open' if opened else ''}><summary>"
        f"<h2>{title}</h2></summary>{content}</details>"
        for key, title, content, opened in parts
    )
    body = f"""<header class=topbar>
<div class=navrow><b>{html.escape(live.label)}</b><nav class=nav>{nav}</nav>
<span id=livebar>{bar}</span></div>
<div id=selbar class=selbar></div>
<script type="application/json" id=seldata>{seldata}</script>
<script>ftSel.init(JSON.parse(document.getElementById("seldata").textContent))</script>
</header>
{rest}"""
    explain = _json(EXPLAIN)
    info_js = (
        f"window.ftExplain={explain};window.ftInfo=function(k){{var "
        "t=window.ftExplain[k];return t?'<span class=info tabindex=0 "
        'data-tip="\'+t.replace(/"/g,"&quot;")+\'">\\u24d8</span>\':""};'
    )
    page = (
        "<!doctype html><html><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        "<title>Fine-tuning dashboard</title>"
        f"<style>{ui_colours()}{_asset('style.css')}</style>"
        f"<script>{info_js}</script>"
        f"<script>{_asset('player.js')}\n{_asset('sel.js')}</script></head>"
        f"<body>{body}<script>{_asset('mode.js')}\n{_asset('nav.js')}\n"
        f"{_asset('expand.js')}\n{'' if ctx.once else _asset('live.js')}"
        "</script></body></html>"
    )
    return page, updates
