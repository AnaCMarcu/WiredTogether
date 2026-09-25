#!/usr/bin/env python
"""Serve the project page locally without letting the browser cache anything.

`python -m http.server` sends `Last-Modified` and no `Cache-Control`, so a
browser is free to reuse a page or script it fetched minutes ago without
asking — which, while the page is being edited, shows stale layouts and
missing buttons that a hard refresh then "fixes". This server marks every
response `no-store`, so each load is the file on disk.

Usage:
  python analysis/serve_site.py               # site/ on http://localhost:8000
  python analysis/serve_site.py --root . --port 8001   # the repo root (review page)
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from paths import REPO  # noqa: E402  (also puts siblings on sys.path)


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args) -> None:   # quiet: one line per request is noise here
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "site")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    root = args.root.resolve()
    handler = partial(NoCacheHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"serving {root} on http://localhost:{args.port}/ (no-store)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
