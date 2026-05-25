"""Built-in axes. Importing this module registers them with the global
axis registry. The package ``__init__`` imports this so axes are
available the moment ``pfnstudio_core`` is imported."""

from . import monotonicity  # noqa: F401  (registers on import)
