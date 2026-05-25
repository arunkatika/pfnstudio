"""Tiny dev server for the rendered site."""

from __future__ import annotations

import http.server
import socketserver
from pathlib import Path


def serve_dir(directory: Path, port: int = 8000) -> None:
    handler_factory = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(directory), **kwargs
    )
    with socketserver.TCPServer(("127.0.0.1", port), handler_factory) as httpd:
        print(f"Serving {directory} at http://127.0.0.1:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")
