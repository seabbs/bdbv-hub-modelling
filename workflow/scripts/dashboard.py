"""Serve the fine-tuning dashboard, or write it once as a static page.

    dashboard.py [--port 8765] [--run NAME ...] [--ref SPEC ...]
    dashboard.py --once FILE [--run NAME ...]

The runs are the workflow's (`runs` in workflow/config.yaml), or those
`--run` names; references come from its `dashboard` section and `--ref`
(see `flowlib.dashboard.context`). Serves on 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig
from flowlib.dashboard import context, nowcasts, page

CONFIG = Path(__file__).parents[1] / "config.yaml"
LOCK = threading.Lock()


def parser() -> argparse.ArgumentParser:
    """Return the command line's parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--once", type=Path, metavar="FILE")
    p.add_argument("--run", action="append", default=[], metavar="NAME")
    p.add_argument("--ref", action="append", default=[], metavar="SPEC")
    p.add_argument("--config", type=Path, default=CONFIG)
    return p


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

    def __init__(self, ctx: page.Context) -> None:
        self.ctx = ctx
        self.html: bytes | None = None
        self.parts: dict[str, str] = {}
        self.builts: dict[str, object] = {}

    def refresh(self) -> None:
        """Build the page once and swap it in."""
        with LOCK:
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


def handler(
    ctx: page.Context, pages: Pages | None = None
) -> type[BaseHTTPRequestHandler]:
    """Return the request handler: the page, its parts and location data.

    They come from `pages`' last build; without `pages`, one is built now.
    """
    if pages is None:
        pages = Pages(ctx)
        pages.rebuild()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/sections.json":
                    query = parse_qs(urlsplit(self.path).query)
                    self._sections(query.get("have", [""])[0])
                elif path.startswith("/data/"):
                    self._data(path.removeprefix("/data/"))
                elif path in ("/", "/index.html"):
                    body = pages.html or BUILDING
                    self._send(200, body, "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                # The client left (a closed tab, a proxy timeout).
                return
            except Exception:
                # The detail goes to the server's log, not the browser.
                traceback.print_exc(file=sys.stderr)
                self._send(500, b"error: see the server's log", "text/plain")

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

        def log_message(self, *_args: object) -> None:
            pass

    return Handler


def write_once(ctx: page.Context, out: Path) -> None:
    """Write the page to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.build(ctx))
    print("wrote", out)


def main() -> None:
    """Serve the dashboard or write it once."""
    a = parser().parse_args()
    config = yaml.safe_load(a.config.read_text())
    ctx = context.build(
        config, localconfig.read(), runs=a.run, refs=a.ref, once=bool(a.once)
    )
    if a.once:
        write_once(ctx, a.once)
        return
    # The page and nowcasts are rebuilt in the background, so no request
    # waits on a build.
    pages = Pages(ctx)
    pages.start(float(config["dashboard"]["refresh_every"]))
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler(ctx, pages))
    print(f"http://127.0.0.1:{a.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
