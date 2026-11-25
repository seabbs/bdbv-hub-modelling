"""Serve the fine-tuning and simulations dashboards, or write one once.

    dashboard.py [--run NAME ...] [--ref SPEC ...] [--while-pid PID]
    dashboard.py --once FILE [--run NAME ...]

The runs are the workflow's (`runs` in workflow/config.yaml), or those
`--run` names; references come from its `dashboard` section and `--ref`
(see `flowlib.dashboard.context`). It listens on the machine settings'
`DASHBOARD_HOST` and `DASHBOARD_PORT` (config/local.env.example), showing
the runs under `DASHBOARD_ROOT` (default `ROOT`) at `/` and their
simulations at `/sims/` (flowlib.dashboard.sims). With `--while-pid` it
ends when that process does: a workflow run serves it so
(flowlib.dashboard.serve).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig
from flowlib.dashboard import context, nowcasts, page, serve, sims, validation
from flowlib.dashboard.runs import Run

CHECKOUT = Path(__file__).parents[2]
CONFIG = CHECKOUT / "workflow" / "config.yaml"
LOCK = threading.Lock()
CLIENT_GONE = (ConnectionError, TimeoutError)
"""How a client that left shows up: a closed tab, a reset, a timeout."""


def parser() -> argparse.ArgumentParser:
    """Return the command line's parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", type=Path, metavar="FILE")
    p.add_argument("--run", action="append", default=[], metavar="NAME")
    p.add_argument("--ref", action="append", default=[], metavar="SPEC")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--while-pid", type=int, metavar="PID")
    return p


def end_with(pid: int, every: float = 30.0) -> None:
    """End this process once process `pid` has ended."""
    from flowlib.dashboard import serve

    def watch() -> None:
        while serve.alive(pid):
            time.sleep(every)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


BUILDING = (
    b"<!doctype html><html><head><meta charset=utf-8>"
    b'<meta http-equiv="refresh" content="10">'
    b"<title>Fine-tuning dashboard</title></head><body>"
    b"<p>Building the dashboard for the first time; this page reloads "
    b"itself.</p></body></html>"
)
"""The page served before the first build has finished."""


class Pages:
    """The last page built, its parts and nowcasts, rebuilt in the background.

    Requests read the last finished build and never wait on one: a build
    (the nowcasts' rebuild can take minutes) holds `LOCK`, a request does
    not.
    """

    def __init__(
        self,
        ctx: page.Context,
        context_of: Callable[[], page.Context] | None = None,
    ) -> None:
        self.ctx = ctx
        # Built afresh on each refresh when set, so a new run shows.
        self.context_of = context_of
        self.html: bytes | None = None
        self.parts: dict[str, str] = {}
        self.builts: dict[str, object] = {}

    def refresh(self) -> None:
        """Build the page once and swap it in."""
        with LOCK:
            if self.context_of is not None:
                self.ctx = self.context_of()
            html, parts = page.render(self.ctx)
            builts = {
                d["key"]: self.ctx.nowcasts(d["key"]) for d in self.ctx.datasets
            }
        self.html, self.parts, self.builts = html.encode(), parts, builts

    def rebuild(self) -> None:
        """Refresh, logging a failed build: the last good page stays."""
        try:
            self.refresh()
        except Exception:
            # The detail goes to the server's log, never to the browser.
            traceback.print_exc(file=sys.stderr)

    def start(self, every: float) -> None:
        """Rebuild every `every` seconds in a background thread."""

        def loop() -> None:
            while True:
                self.rebuild()
                time.sleep(every)

        threading.Thread(target=loop, daemon=True).start()


class _Sending(BaseHTTPRequestHandler):
    """A handler that sends whole bodies and logs nothing."""

    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


JSON = "application/json"
NOT_FOUND = (404, b"{}", JSON)


def sims_answer(
    index: sims.Index, layer_of: dict[str, str], path: str
) -> tuple[int, bytes, str]:
    """Return the answer to a request under /sims/.

    The page (`/sims/`), the sources (`/sims/sources.json`), a source's
    index (`/sims/index/<source>.json`) and one simulation
    (`/sims/sim/<source>/<id>.json`).
    """
    rest = path.removeprefix("/sims")
    if rest in ("", "/", "/index.html"):
        return 200, sims.page().encode(), "text/html; charset=utf-8"
    if rest == "/sources.json":
        return 200, page._json(index.listing()).encode(), JSON
    if rest.startswith("/index/") and rest.endswith(".json"):
        body = index.encoded_data(rest[len("/index/") : -len(".json")])
        return (200, body, JSON) if body is not None else NOT_FOUND
    if rest.startswith("/sim/") and rest.endswith(".json"):
        source, _, oid = rest[len("/sim/") : -len(".json")].partition("/")
        folder = index.sources().get(source)
        found = sims.simulation(folder, oid, layer_of) if folder else None
        if found is None:
            return NOT_FOUND
        return 200, page._json(found).encode(), JSON
    return 404, b"not found", "text/plain"


def sims_handler(
    index: sims.Index, layer_of: dict[str, str]
) -> type[BaseHTTPRequestHandler]:
    """Return a request handler for the simulations alone."""

    class Handler(_Sending):
        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/sims" or path.startswith("/sims/"):
                    self._send(*sims_answer(index, layer_of, path))
                else:
                    self._send(404, b"not found", "text/plain")
            except CLIENT_GONE:
                return

    return Handler


def validation_answer(
    run_list: Sequence[Run], path: str
) -> tuple[int, bytes, str]:
    """Return the answer to a request under /validation/.

    A run's validation forecasts' models and blocks
    (`/validation/<run>/index.json`) and one block
    (`/validation/<run>/<block>.json`), from `flowlib.dashboard.validation`.
    """
    run_key, _, name = path.removeprefix("/validation/").partition("/")
    run = next((r for r in run_list if r.key == run_key), None)
    if run is None or not name.endswith(".json"):
        return NOT_FOUND
    name = name.removesuffix(".json")
    if name == "index":
        return 200, page._json(validation.index(run)).encode(), JSON
    if not name.isdigit():
        return NOT_FOUND
    found = validation.block(run, int(name))
    if found is None:
        return NOT_FOUND
    return 200, page._json(found).encode(), JSON


def validation_handler(
    run_list: Sequence[Run],
) -> type[BaseHTTPRequestHandler]:
    """Return a request handler for the validation forecasts alone."""

    class Handler(_Sending):
        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            try:
                if path.startswith("/validation/"):
                    self._send(*validation_answer(run_list, path))
                else:
                    self._send(404, b"not found", "text/plain")
            except CLIENT_GONE:
                return

    return Handler


def handler(
    ctx: page.Context,
    pages: Pages | None = None,
    simulations: tuple[sims.Index, dict[str, str]] | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Return the request handler: the page, its parts and location data.

    They come from `pages`' last build; without `pages`, one is built now.
    With `simulations` (an index and the grid's layers) it answers
    /sims/ too (`sims_answer`).
    """
    if pages is None:
        pages = Pages(ctx)
        pages.rebuild()

    class Handler(_Sending):
        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            try:
                sims_path = path == "/sims" or path.startswith("/sims/")
                if path.startswith("/validation/"):
                    # The runs of the last build: a new run shows.
                    self._send(*validation_answer(pages.ctx.runs, path))
                elif simulations is not None and sims_path:
                    self._send(*sims_answer(*simulations, path))
                elif path == "/sections.json":
                    query = parse_qs(urlsplit(self.path).query)
                    self._sections(query.get("have", [""])[0])
                elif path.startswith("/data/"):
                    self._data(path.removeprefix("/data/"))
                elif path in ("/", "/index.html"):
                    body = pages.html or BUILDING
                    self._send(200, body, "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")
            except CLIENT_GONE:
                # The client left (a closed tab, a timeout).
                return
            except Exception:
                # The detail goes to the server's log, not the browser.
                traceback.print_exc(file=sys.stderr)
                try:
                    self._send(
                        500, b"error: see the server's log", "text/plain"
                    )
                except CLIENT_GONE:
                    return

        def _sections(self, have: str) -> None:
            # Each part's hash, with its html only where `have` (key:hash,
            # comma separated) holds another hash or none.
            known = dict(
                item.split(":", 1) for item in have.split(",") if ":" in item
            )
            parts = pages.parts
            out = {}
            for key, content in parts.items():
                digest = hashlib.sha1(content.encode()).hexdigest()[:12]
                out[key] = (
                    {"hash": digest}
                    if known.get(key) == digest
                    else {"hash": digest, "html": content}
                )
            self._send(200, page._json(out).encode(), "application/json")

        def _data(self, name: str) -> None:
            # /data/<dataset>/<location>.json
            key, _, loc = name.removesuffix(".json").partition("/")
            if key not in {d["key"] for d in ctx.datasets} or not loc:
                self._send(404, b"{}", "application/json")
                return
            built = pages.builts.get(key)
            found = nowcasts.location(built, loc) if built else None
            if not found:
                self._send(404, b"{}", "application/json")
                return
            self._send(200, page._json(found).encode(), "application/json")

    return Handler


def address(local: dict[str, str]) -> tuple[str, int]:
    """Return the host and port the machine settings serve the page on."""
    return local["DASHBOARD_HOST"], int(local["DASHBOARD_PORT"])


def write_once(ctx: page.Context, out: Path) -> None:
    """Write the page to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.build(ctx))
    print("wrote", out)


def hidden_rounds(config: dict) -> set[int]:
    """Return the rounds the dashboard never shows: the grid's test round.

    It is held out for the paper (`grid_rounds.test_round`).
    """
    test = (config.get("grid_rounds") or {}).get("test_round")
    return set() if test is None else {int(test)}


def simulations(
    config: dict, local: dict[str, str], every: float
) -> tuple[sims.Index, dict[str, str]]:
    """Return the simulations' index, refreshed every `every` seconds.

    Of the simulations under DASHBOARD_ROOT, each batch's summary cached
    in `<DASHBOARD_ROOT>/dashboard/sims/`, with the layers of the grid
    the workflow simulates (`simulate.grid_file`). The grid's test round
    is never listed (`hidden_rounds`).
    """
    layer_of = sims.layers(CHECKOUT / config["simulate"]["grid_file"])
    root = serve.root(local)
    index = sims.Index(
        root, root / "dashboard" / "sims", layer_of, hidden_rounds(config)
    )

    def loop() -> None:
        while True:
            try:
                index.refresh()
            except Exception:
                traceback.print_exc(file=sys.stderr)
            time.sleep(every)

    threading.Thread(target=loop, daemon=True).start()
    return index, layer_of


def main() -> None:
    """Serve the dashboard or write it once."""
    a = parser().parse_args()
    config = yaml.safe_load(a.config.read_text())
    localconfig.check()
    local = localconfig.read()
    ctx = context.build(
        config, local, runs=a.run, refs=a.ref, once=bool(a.once)
    )
    if a.once:
        write_once(ctx, a.once)
        return
    # The page and nowcasts are rebuilt in the background, so no request
    # waits on a build.
    if a.while_pid:
        end_with(a.while_pid)
    # With no --run, the runs under ROOT are found again on each refresh.
    pages = Pages(
        ctx,
        None if a.run else lambda: context.build(config, local, refs=a.ref),
    )
    every = float(config["dashboard"]["refresh_every"])
    pages.start(every)
    host, port = address(local)
    server = ThreadingHTTPServer(
        (host, port),
        handler(ctx, pages, simulations(config, local, every)),
    )
    print(f"http://{host}:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
