"""Long-lived in-process inference worker.

A serve() entry point loads a trained checkpoint via
:class:`pfnstudio_core.training.predict.ModelLoader` and exposes it as
a tiny HTTP server on a loopback port. Studio's NestJS process spawns
one of these per Deployment and proxies user requests to it.

Stdlib http.server (ThreadingHTTPServer) is used instead of FastAPI to
keep the core install lightweight — no extra runtime deps. Concurrency
is bounded by the GIL during the model's forward pass anyway, so the
threading model is fine for Phase 1's load profile.
"""

from .hf import pull_snapshot, push_folder
from .worker import serve

__all__ = ["serve", "push_folder", "pull_snapshot"]
